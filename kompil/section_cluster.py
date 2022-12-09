import os
import cv2
import torch
import decord
import contextlib

import plotly.io
import plotly.graph_objects as go

from typing import Optional
from tqdm import tqdm

from airium import Airium
from kornia.feature.siftdesc import SIFTDescriptor
from kornia.color import bgr_to_grayscale

from kompil.utils.resources import get_video
from kompil.utils.video import tensor_to_numpy, decord_to_tensor
from kompil.utils.kmeans import kmeans
from kompil.nn.topology.builder import AutoFlow
from kompil.cli_defaults import SectionClusterDefaults as defaults


def _load_data_from_autoflow_file(autoflow_fpath: str) -> torch.Tensor:
    assert autoflow_fpath and os.path.exists(autoflow_fpath)

    autoflow_data = torch.load(autoflow_fpath)

    topology_json = autoflow_data["model_meta_dict"]["topology"]

    assert topology_json, f"Could not get topology data of the registered layer at {autoflow_fpath}"

    assert len(topology_json) == 1  # must contains only 1 layer

    autoflow_data_json = topology_json[0]
    autoflow_type = autoflow_data_json["type"]

    assert autoflow_type == AutoFlow.TYPE  # this layer must be autoflow

    autoflow_tensors_dict = autoflow_data["model_state_dict"]

    assert autoflow_tensors_dict

    # converts to list and get the only one value (ie : a tensor)
    data = list(autoflow_tensors_dict.values())[0]

    # Flatten because of kmeans implem restrictions
    data = torch.flatten(data, start_dim=1, end_dim=-1)

    return data


def _get_video_sift_desc(video_fpath: str) -> torch.Tensor:
    size = 120
    ang = 10
    spac = 10
    batch = 1024

    tqdm_meter = tqdm(desc="[running SIFT]")
    sift = SIFTDescriptor(size, ang, spac)
    video = decord.VideoReader(
        video_fpath, ctx=decord.cpu(), num_threads=0, height=size, width=size
    )
    nb_frame = len(video)
    frame_data = torch.empty(size=(nb_frame, ang * spac**2))

    for i in range(0, nb_frame, batch):
        if i + batch >= nb_frame:
            batch = nb_frame - i

        frame = video[i : (i + batch)]
        frame = frame.permute(0, 3, 1, 2)  # bhwc to bchw
        gray_resize = bgr_to_grayscale(frame) / 255.0
        sift_res = sift(gray_resize).detach()
        frame_data[i : (i + batch)] = sift_res

        tqdm_meter.set_postfix(frames=f"{i+batch}")
        tqdm_meter.update()

    return frame_data


def _pca(data: torch.Tensor, nb_component: int) -> torch.Tensor:
    assert nb_component > 0

    _, _, v = torch.pca_lowrank(data, q=nb_component)

    components = torch.matmul(data, v[:, :nb_component])

    return components


def gen_data(video_fpath: str, data_fpath: str, method: str = defaults.GEN_METHOD) -> torch.Tensor:
    video_fpath = get_video(video_fpath)

    method = method if method else defaults.GEN_METHOD

    if method == "sift":
        data = _get_video_sift_desc(video_fpath)
    elif method == "sift-pca":
        sift_desc = _get_video_sift_desc(video_fpath)
        nb_component = max(24, int(sift_desc.shape[1] / 10))
        data = _pca(sift_desc, nb_component)
    elif method == "autoflow":
        from kompil.encode import encode
        from kompil.nn.layers.save_load import save_save_layers

        af_fpath = "build/layers/af.pth"

        final_model, _ = encode(
            model_name="tmp_cluster",
            video_name=video_fpath,
            learning_rate="primus",
            loading_mode="full-ram",
            scheduler="tycho",
            batch_size=16,
            topology_builder="find_clusters_simple",
            model_extra=[af_fpath],
            resolution="100p",
            autoquit_epoch=300,
            precision=16,
        )
        save_save_layers(final_model)
        data = _load_data_from_autoflow_file(af_fpath)
    else:
        raise NotImplementedError(method)

    if data_fpath:
        torch.save(data, data_fpath)

    return data


def _gather_similar_frames(
    frame_data: torch.Tensor, cluster_idx: torch.Tensor, epochs: int
) -> torch.Tensor:
    similarity_dist = 0.3
    topk = 250
    tqdm_meter = tqdm(desc="[running FIX]")

    for e in range(epochs):
        new_cluster_idx = cluster_idx.clone()

        for i in range(1, len(cluster_idx)):
            a = i - 1
            b = i

            a_frame = frame_data[a]
            b_frame = frame_data[b]

            a_frame_cluster_idx = cluster_idx[a]
            b_frame_cluster_idx = cluster_idx[b]

            dist_frame = torch.sqrt(torch.sum(torch.square(a_frame - b_frame)))

            # Frame very similar but not is same cluster...
            if dist_frame < similarity_dist and a_frame_cluster_idx != b_frame_cluster_idx:
                # Get frames of the same cluster of frame a and b
                a_cluster = frame_data[torch.nonzero(cluster_idx == a_frame_cluster_idx)].squeeze(1)
                b_cluster = frame_data[torch.nonzero(cluster_idx == b_frame_cluster_idx)].squeeze(1)

                # To avoid out of bound k
                fixed_topk = min(topk, a_cluster.shape[0], b_cluster.shape[0])

                # Distance between frame A and nearest K frames inside cluster A
                aa_closest_frames_dist, _ = torch.topk(
                    torch.sqrt(torch.sum(torch.square(a_cluster - a_frame), dim=1)),
                    fixed_topk,
                    largest=False,
                )
                # Distance between frame A and nearest K frames inside cluster B
                ab_closest_frames_dist, _ = torch.topk(
                    torch.sqrt(torch.sum(torch.square(b_cluster - a_frame), dim=1)),
                    fixed_topk,
                    largest=False,
                )
                # Distance between frame B and nearest K frames inside cluster B
                bb_closest_frames_dist, _ = torch.topk(
                    torch.sqrt(torch.sum(torch.square(b_cluster - b_frame), dim=1)),
                    fixed_topk,
                    largest=False,
                )
                # Distance between frame B and nearest K frames inside cluster A
                ba_closest_frames_dist, _ = torch.topk(
                    torch.sqrt(torch.sum(torch.square(a_cluster - b_frame), dim=1)),
                    fixed_topk,
                    largest=False,
                )

                # For now we compare only mean, maybe [std, min, max] can be useful also
                aa_mean = aa_closest_frames_dist.mean()
                ab_mean = ab_closest_frames_dist.mean()
                bb_mean = bb_closest_frames_dist.mean()
                ba_mean = ba_closest_frames_dist.mean()

                # Dist between frame B and cluster A is closest than frame B and cluster B
                if ba_mean < bb_mean + 0.01:
                    new_cluster_idx[b] = a_frame_cluster_idx

                # Dist between frame A and cluster B is closest than frame A and cluster A
                if ab_mean < aa_mean + 0.01:
                    new_cluster_idx[a] = b_frame_cluster_idx

        # No change, stop
        if torch.allclose(new_cluster_idx, cluster_idx):
            break

        # Just print
        dist = torch.sqrt(torch.sum(torch.square(new_cluster_idx - cluster_idx)))
        tqdm_meter.set_postfix(iteration=f"{e}", frame_shift=f"{dist:0.2f}")
        tqdm_meter.update()

        # Reduce top K frame to focus remaining frame to match with very-close-frames in cluster
        topk = max(20, topk - 10)
        cluster_idx = new_cluster_idx

    return new_cluster_idx


def find(
    data_file: str, nb_cluster: int, output_file: str, threshold: float = defaults.THRESHOLD
) -> torch.Tensor:
    threshold = threshold if threshold else defaults.THRESHOLD

    frame_data = torch.load(data_file)

    cluster_idx, _ = kmeans(x=frame_data, k=nb_cluster, epochs=50, tol=threshold)
    cluster_idx = _gather_similar_frames(frame_data, cluster_idx, epochs=50)

    if output_file:
        torch.save(cluster_idx, output_file)

    return cluster_idx


def _analysis_frame_dist(
    t_autoflow_data: torch.Tensor, t_cluster_idx: torch.Tensor, cluster_means: torch.Tensor
):
    if t_autoflow_data is not None:
        frame_mean = torch.index_select(cluster_means, 0, t_cluster_idx)
        euclidian = torch.sum(torch.square(t_autoflow_data - frame_mean), dim=1)
    else:
        euclidian = torch.ones_like(t_cluster_idx)
    bar_obj = go.Bar(y=euclidian.cpu(), marker_color=t_cluster_idx.cpu())
    return go.Figure(data=[bar_obj])


def _analysis_cluster_histogram(t_cluster_idx: torch.Tensor):
    return go.Figure(data=[go.Histogram(x=t_cluster_idx.cpu())])


def _analysis_cluster_segments(t_counts: torch.Tensor, t_mask_id: torch.Tensor):
    t_counts_segment = torch.masked_select(t_counts, t_mask_id)
    bar_obj = go.Bar(y=t_counts_segment.cpu())
    return go.Figure(data=[bar_obj])


def _find_cluster_segments(
    t_cons_counts: torch.Tensor, t_cluster_cons_ids: torch.Tensor, cluster_id: int
):
    t_mask_id = t_cluster_cons_ids == cluster_id
    t_frame_ids = torch.cumsum(t_cons_counts, 0)
    t_frame_ids[1:].copy_(t_frame_ids[:-1].clone())
    t_frame_ids[0] = 0
    t_frame_counts = torch.masked_select(t_cons_counts, t_mask_id)
    separations = torch.masked_select(t_frame_ids, t_mask_id)
    return torch.stack([separations, t_frame_counts], dim=1).cpu().numpy()


@contextlib.contextmanager
def html_page(a: Airium, cluster_count: int) -> Airium:
    a("<!DOCTYPE html>")
    with a.html(lang="pl"):
        with a.head():
            a.meta(charset="utf-8")
            a.title(_t="Cluster study")

        with a.body():
            with a.table():
                with a.tr():
                    with a.td(valign="top", style="min-width:150px"):

                        with a.a(href=f"index.html"):
                            a("Index")
                        a.br()

                        for cluster_id in range(cluster_count):
                            with a.a(href=f"cluster_{cluster_id}.html"):
                                a(f"Cluster {cluster_id}")
                            a.br()

                    with a.td(valign="top", style="width:100%"):
                        yield


class PageBuilder:
    def __init__(self, t_cluster_idx: torch.Tensor, t_data: Optional[torch.Tensor]) -> None:
        self.t_cluster_idx = t_cluster_idx
        self.t_data = t_data
        self.cluster_count = t_cluster_idx.max() + 1
        self.frames_count = t_cluster_idx.numel()
        self.frames_visited = []
        # Clusters calculations
        res = torch.unique_consecutive(self.t_cluster_idx, return_counts=True)
        self.t_cluster_consecutive_ids, self.t_cluster_consecutive_counts = res
        # Segments calculations
        self.t_cluster_segment_count = torch.zeros(
            self.cluster_count, dtype=torch.int, device=t_cluster_idx.device
        )
        for cluster_id in range(self.cluster_count):
            t_mask_id = self.t_cluster_consecutive_ids == cluster_id
            self.t_cluster_segment_count[cluster_id] = t_mask_id.int().sum()
        # Recalculate cluster means
        if t_data is not None:
            self.t_cluster_means = self.__init_cluster_means(t_cluster_idx, t_data)
            self.latent_size = t_data.shape[1]

    @staticmethod
    def __init_cluster_means(t_cluster_idx: torch.Tensor, t_data: torch.Tensor) -> torch.Tensor:
        frames_count, latent_size = t_data.shape
        cluster_count = t_cluster_idx.max() + 1
        t_cluster_means = torch.empty(cluster_count, latent_size, device=t_data.device)
        for cluster in range(cluster_count):
            t_cluster_mask = t_cluster_idx == cluster
            t_cluster_mask = t_cluster_mask.repeat_interleave(latent_size)
            t_cluster_mask = t_cluster_mask.view(frames_count, latent_size)
            t_values = torch.masked_select(t_data, t_cluster_mask).view(-1, latent_size)
            t_cluster_means[cluster].copy_(t_values.mean(dim=0))

        return t_cluster_means

    @property
    def has_data(self):
        return self.t_data is not None

    def __add_frame(self, a, frame_id):
        if frame_id >= self.frames_count or frame_id < 0:
            return
        else:
            a.img(src=os.path.join("frames", f"frame_{frame_id}.jpg"))
            self.frames_visited.append(frame_id)

    def build_index(self) -> str:
        # Build figures
        if self.has_data:
            fig_frame_dist = _analysis_frame_dist(
                self.t_data, self.t_cluster_idx, self.t_cluster_means
            )
            fig_histogram = _analysis_cluster_histogram(self.t_cluster_idx)

        # Build page
        a = Airium()
        with html_page(a, self.cluster_count):
            with a.h3():
                a("Index page")

            if self.has_data:
                with a.h4():
                    a("Frame distances to its mean")
                a(plotly.io.to_html(fig_frame_dist))
                with a.h4():
                    a("Frames count per clusters")
                a(plotly.io.to_html(fig_histogram))
            else:
                a("Lacking data")

        return str(a)

    def build_cluster(self, cluster_id: int) -> str:
        # Calculations
        t_counts = self.t_cluster_consecutive_counts
        t_consids = self.t_cluster_consecutive_ids
        t_mask_id = t_consids == cluster_id
        segments = _find_cluster_segments(t_counts, t_consids, cluster_id)

        # Build page
        a = Airium()
        with html_page(a, self.cluster_count):
            with a.h3():
                a(f"Cluster {cluster_id}")
            # Frame count bars
            with a.h4():
                a("Number of frames per sections")
            a(plotly.io.to_html(_analysis_cluster_segments(t_counts, t_mask_id)))
            # Image table
            with a.h4():
                a(f"Sections details")
            with a.table(border=1):
                with a.tr():
                    a.th(_t="no.")
                    a.th(_t="Start (included)")
                    a.th(_t="End (included)")
                    a.th(_t="Count")
                    a.th(_t="First - 1 (excluded)")
                    a.th(_t="First (included)")
                    a.th(_t="Last (included)")
                    a.th(_t="Last + 1 (excluded)")
                    a.th(_t="")

                for segment_id, (start_frame, frame_count) in enumerate(segments):
                    with a.tr():
                        a.td(_t=str(segment_id))
                        a.td(_t=str(start_frame))
                        a.td(_t=str(start_frame + frame_count - 1))
                        a.td(_t=str(frame_count))
                        with a.td():
                            self.__add_frame(a, start_frame - 1)
                        with a.td():
                            self.__add_frame(a, start_frame)
                        with a.td():
                            self.__add_frame(a, start_frame + frame_count - 1)
                        with a.td():
                            self.__add_frame(a, start_frame + frame_count)
                        with a.td():
                            with a.a(href=f"segment_{cluster_id}_{segment_id}.html"):
                                a(f"link to segment page")

        return str(a)

    def __build_segment_figures(self, cluster_id: int, first_frame: int, last_frame: int):
        # No data, no figures
        if not self.has_data:
            return None, None

        # Figures
        ls_cluster = self.t_cluster_means[cluster_id]
        ls_cluster, sort_index = torch.sort(ls_cluster)
        fig_start = None
        if first_frame > 0:
            # Isolate data
            ls_start_frame = self.t_data[first_frame]
            ls_before_frame = self.t_data[first_frame - 1]

            # Sort data
            ls_start_frame = torch.take(ls_start_frame, sort_index)
            ls_before_frame = torch.take(ls_before_frame, sort_index)

            # Build figure
            fig_start = go.Figure()
            fig_start.add_trace(go.Scatter(y=ls_start_frame.cpu(), name="first frame"))
            fig_start.add_trace(go.Scatter(y=ls_before_frame.cpu(), name="before frame"))
            fig_start.add_trace(go.Scatter(y=ls_cluster.cpu(), name="cluster mean"))

        fig_end = None
        if last_frame < self.frames_count - 1:
            # Isolate data
            ls_last_frame = self.t_data[last_frame]
            ls_after_frame = self.t_data[last_frame + 1]

            # Sort data
            ls_last_frame = torch.take(ls_last_frame, sort_index)
            ls_after_frame = torch.take(ls_after_frame, sort_index)

            # Build figure
            fig_end = go.Figure()
            fig_end.add_trace(go.Scatter(y=ls_last_frame.cpu(), name="last frame"))
            fig_end.add_trace(go.Scatter(y=ls_after_frame.cpu(), name="after frame"))
            fig_end.add_trace(go.Scatter(y=ls_cluster.cpu(), name="cluster mean"))

        # return figures
        return fig_start, fig_end

    def build_segment(self, cluster_id: int, segment_id: int) -> str:
        # Find concerned frames
        t_counts = self.t_cluster_consecutive_counts
        t_consids = self.t_cluster_consecutive_ids
        segments = _find_cluster_segments(t_counts, t_consids, cluster_id)
        first_frame, frame_count = segments[segment_id]
        last_frame = first_frame + frame_count - 1

        # Figures
        fig_start, fig_end = self.__build_segment_figures(cluster_id, first_frame, last_frame)

        # Build page
        a = Airium()
        with html_page(a, self.cluster_count):
            with a.h3():
                a(f"Segment {cluster_id}:{segment_id}")
            start_frame, frame_count = segments[segment_id]
            a(
                f"Segment between frame {start_frame} and "
                f"frame {start_frame + frame_count - 1} included."
            )

            with a.table(border=1):
                with a.tr():
                    a.th(_t="First - 1 (excluded)")
                    a.th(_t="First (included)")
                    a.th(_t="Last (included)")
                    a.th(_t="Last + 1 (excluded)")

                with a.tr():
                    with a.td():
                        self.__add_frame(a, first_frame - 1)
                    with a.td():
                        self.__add_frame(a, first_frame)
                    with a.td():
                        self.__add_frame(a, last_frame)
                    with a.td():
                        self.__add_frame(a, last_frame + 1)

            if fig_start is not None:
                a(plotly.io.to_html(fig_start))
            if fig_end is not None:
                a(plotly.io.to_html(fig_end))

        return str(a)


def cluster_report(cluster_fpath: str, video_name: str, output: Optional[str], data: Optional[str]):
    # Read value from disk
    t_cluster_idx = torch.load(cluster_fpath)
    t_data = torch.load(data) if data is not None else None
    video_fpath = get_video(video_name)

    # To device
    device = torch.device("cuda")  # hardcoded for now
    t_cluster_idx = t_cluster_idx.to(device)
    cluster_count = t_cluster_idx.max() + 1
    t_data = t_data.to(device)

    # Builder
    buidler = PageBuilder(t_cluster_idx, t_data)

    # Init folder
    root_folder = "/tmp/kompil_cluster_report" if output is None else os.path.abspath(output)
    os.makedirs(root_folder, exist_ok=True)

    # Keep video
    reported_video_fpath = os.path.join(root_folder, "video")
    same_video = False
    if os.path.exists(reported_video_fpath):
        with open(reported_video_fpath, "r") as f:
            same_video = f.read() == video_name

    with open(reported_video_fpath, "w+") as f:
        f.write(video_name)

    # Write index
    index_fpath = os.path.join(root_folder, "index.html")

    index_html = buidler.build_index()
    with open(index_fpath, "w+") as f:
        f.write(index_html)

    # Write clusters
    for cluster_id in range(buidler.cluster_count):
        cluster_html = buidler.build_cluster(cluster_id)
        cluster_fpath = os.path.join(root_folder, f"cluster_{cluster_id}.html")
        with open(cluster_fpath, "w+") as f:
            f.write(cluster_html)

    # Write segments
    for cluster_id in range(buidler.cluster_count):
        for segment_id in range(buidler.t_cluster_segment_count[cluster_id]):
            segment_html = buidler.build_segment(cluster_id, segment_id)
            fpath = os.path.join(root_folder, f"segment_{cluster_id}_{segment_id}.html")
            with open(fpath, "w+") as f:
                f.write(segment_html)

    # Write frames
    buidler.frames_visited.sort()  # to go faster
    video = decord.VideoReader(video_fpath, ctx=decord.cpu(), num_threads=0, height=160, width=240)
    frames_folder = os.path.join(root_folder, "frames")
    os.makedirs(frames_folder, exist_ok=True)
    written_frames = []
    for frame_id in buidler.frames_visited:
        if frame_id in written_frames:
            continue
        img = tensor_to_numpy(decord_to_tensor(video[frame_id]))
        imgpath = os.path.join(frames_folder, f"frame_{frame_id}.jpg")
        if not same_video or not os.path.exists(imgpath):
            cv2.imwrite(imgpath, img)
        written_frames.append(frame_id)

    # Indicate link
    print(f"Index written in: file://{index_fpath}")
