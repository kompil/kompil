#!/bin/env python3

import os
import json

import kompil.train.loss.base
import kompil.train.learning_rate
import kompil.nn.topology.pattern
import kompil.train.optimizers.scheduler
import kompil.train.optimizers.optimizer
import kompil.metrics.metrics
import kompil.maskmakers.factory
import kompil.quantize
import kompil.packers
from kompil.utils.video import RESOLUTION_MAP
from kompil.utils.colorspace import COLORSPACE_LIST


PATH = os.path.join(os.path.dirname(__file__), "elements", "run.schema.json")


def template(
    resolutions,
    schedulers,
    optimizers,
    learning_rates,
    losses,
    colorspaces,
    builders,
    quality_metric,
    maskmakers,
    quant_methods,
    packers,
):
    return {
        "definitions": {
            "loss": {"type": "string", "enum": losses},
            "scheduler": {"type": "string", "enum": schedulers},
            "plus_params": {
                "additionalProperties": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "integer"},
                        {"type": "number"},
                        {"type": "boolean"},
                    ]
                }
            },
        },
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "description": "Schema guide to build runs for the sota",
        "type": "object",
        "additionalProperties": False,
        "required": ["group", "name", "video"],
        "properties": {
            "$schema": {"type": "string"},
            "group": {
                "title": "Group name",
                "description": "Common name to regroup the run on",
                "type": "string",
            },
            "name": {
                "title": "Name of the run",
                "description": "Name of this specific run, must be unique.",
                "type": "string",
            },
            "video": {
                "title": "Video",
                "description": "Video to encode",
                "type": "string",
            },
            "resolution": {
                "title": "Resolution",
                "description": "Video to encode",
                "type": "string",
                "enum": resolutions,
            },
            "colorspace": {
                "title": "Colorspace",
                "description": "What is the colorspace of the model's output",
                "type": "string",
                "enum": colorspaces,
            },
            "quality_metric": {
                "title": "Quality metric",
                "description": "Which is the quality metric for the take best option.",
                "type": "string",
                "enum": quality_metric,
            },
            "maskmaker": {
                "title": "Maskmaker",
                "description": "Mask generation properties",
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["id", "type"],
                    "properties": {
                        "id": {"title": "Name", "type": "string"},
                        "type": {"type": "string", "enum": maskmakers},
                        "$ref": "#/definitions/plus_params",
                    },
                },
            },
            "training": {
                "title": "Everything about the training part",
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "topology": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "builder": {
                                "title": "Topology builder",
                                "description": "Builder that will define the topology based on parameters",
                                "type": "string",
                                "enum": builders,
                            },
                            "parameters": {
                                "title": "Model extra",
                                "description": "Parameters for the topology builder",
                                "type": "array",
                                "items": {
                                    "oneOf": [
                                        {"type": "string"},
                                        {"type": "integer"},
                                        {"type": "number"},
                                    ]
                                },
                            },
                        },
                    },
                    "precision": {
                        "title": "Precision",
                        "description": "Run in 16 or 32 bits floats",
                        "type": "integer",
                        "enum": [16, 32],
                    },
                    "batch_size": {
                        "title": "Batch size",
                        "description": "Number of example in one batch",
                        "type": "integer",
                        "minimum": 1,
                    },
                    "max_epochs": {
                        "title": "Maximum epochs",
                        "description": "Number of epochs to learn",
                        "type": "integer",
                        "minimum": 1,
                    },
                    "fine_tuning": {
                        "title": "Fine tuning",
                        "description": "Number of epochs to fine tune the model at the end",
                        "type": "integer",
                        "minimum": 1,
                    },
                    "scheduler": {
                        "title": "Scheduler",
                        "description": "Which scheduler to use",
                        "oneOf": [
                            {
                                "$ref": "#/definitions/scheduler",
                            },
                            {
                                "type": "object",
                                "required": ["name"],
                                "properties": {
                                    "name": {
                                        "title": "Name of the scheduler",
                                        "$ref": "#/definitions/scheduler",
                                    },
                                    "learning_rate": {
                                        "title": "Learning rate",
                                        "description": "Defines the base learning rate for the scheduler",
                                        "oneOf": [
                                            {"type": "string", "enum": learning_rates},
                                            {"type": "number"},
                                        ],
                                    },
                                },
                                "$ref": "#/definitions/plus_params",
                            },
                        ],
                    },
                    "optimizer": {
                        "title": "Optimizer",
                        "description": "Which optimizer to use",
                        "type": "string",
                        "enum": optimizers,
                    },
                    "loss": {
                        "title": "Loss",
                        "description": "Which loss to use",
                        "oneOf": [
                            {
                                "$ref": "#/definitions/loss",
                            },
                            {
                                "type": "object",
                                "required": ["name"],
                                "properties": {
                                    "name": {
                                        "title": "Name of the loss",
                                        "$ref": "#/definitions/loss",
                                    }
                                },
                                "$ref": "#/definitions/plus_params",
                            },
                        ],
                    },
                    "take_best": {
                        "title": "Take best",
                        "description": "The final model will be the best model found instead of the last.",
                        "type": "boolean",
                    },
                    "gradient_clipping": {
                        "title": "Gradient clipping",
                        "description": "Gradient clipping threshold.",
                        "type": "number",
                        "minimum": 0.0,
                    },
                    "accumulate_batches": {
                        "title": "Accumulate batches",
                        "description": "Accumulate gradients of batches to simulate higher batch sizes.",
                        "type": "integer",
                        "minimum": 0,
                    },
                },
            },
            "quantization": {
                "title": "Quantization",
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "method": {
                        "title": "Quantization method",
                        "type": "string",
                        "enum": quant_methods,
                    },
                },
            },
            "packer": {
                "title": "Packer",
                "description": "Packer's name",
                "type": "string",
                "enum": packers,
            },
        },
    }


def main():
    resolutions = ["-1", *RESOLUTION_MAP.keys()]
    schedulers = list(kompil.train.optimizers.scheduler.factory().keys())
    optimizers = list(kompil.train.optimizers.optimizer.factory().keys())
    learning_rates = list(kompil.train.learning_rate.factory().keys())
    losses = list(kompil.train.loss.base.factory().keys())
    colorspaces = COLORSPACE_LIST
    topology_builders = list(kompil.nn.topology.pattern.factory().keys())
    quality_metric = list(kompil.metrics.metrics.factory().keys())
    maskmakers = list(kompil.maskmakers.factory.maskmaker_factory().keys())
    quant_methods = list(kompil.quant.factory().keys())
    packers = list(kompil.packers.factory_packer().keys())

    schema = template(
        resolutions=resolutions,
        schedulers=schedulers,
        optimizers=optimizers,
        learning_rates=learning_rates,
        losses=losses,
        colorspaces=colorspaces,
        builders=topology_builders,
        quality_metric=quality_metric,
        maskmakers=maskmakers,
        quant_methods=quant_methods,
        packers=packers,
    )

    with open(PATH, "w+") as f:
        f.write(json.dumps(schema, indent=4, sort_keys=True))


if __name__ == "__main__":
    main()
