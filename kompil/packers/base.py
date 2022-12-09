from kompil.packers.factory import get_packer, get_unpacker


def pack_model_data(data: dict, packer: str) -> dict:
    packer_fct, unpacker_name, unpacker_version = get_packer(packer)

    assert "quantization" in data, "Packers can only pack a quantized model."

    return {
        **packer_fct(data),
        "unpacker": (unpacker_name, unpacker_version),
    }


def unpack_model_data(data: dict) -> dict:
    assert "unpacker" in data, "Packers can only pack a quantized model"

    unpacker_fct = get_unpacker(*data["unpacker"])

    data.pop("unpacker")

    return unpacker_fct(data)
