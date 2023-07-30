
"""
This is a python script that contains the model and the functions to extract the features from the model, along with the label map.
"""

import torch
from torchvision.ops import boxes as box_ops, roi_align
from torch import Tensor
from typing import Optional, List


def get_features(name, features): 
    """Define a function that will save the output of a layer using a hook"""
    # https://kozodoi.me/blog/20210527/extracting-features
    def hook(model, input, output):
        features[name] = output.detach()

    return hook


class CustomRCNN:
    def __init__(
        self,
        model: torch.nn.Module,
        embedding_layer_name: str,
        embedding_layer: torch.nn.Module,
    ):
        self.model = model

        self.embedding_layer_name = embedding_layer_name
        self._embedding_output = {}  # must be a dictionary
        embedding_layer.register_forward_hook(
            get_features(self.embedding_layer_name, self._embedding_output)
        )

    def __call__(
        # https://github.com/pytorch/vision/blob/3966f9558bfc8443fc4fe16538b33805dd42812d/torchvision/models/detection/generalized_rcnn.py#L46C12-L46C12
        self,
        input_data: list[torch.Tensor],
    ) -> tuple[list[dict], list[torch.Tensor]]:
        batch_size = len(input_data)
        self.model.eval()
        with torch.no_grad():
            original_image_sizes = []
            for img in input_data:
                val = img.shape[-2:]
                original_image_sizes.append((val[0], val[1]))

            images, _ = self.model.transform(input_data)
            features = self.model.backbone(images.tensors)
            proposals, _ = self.model.rpn(images, features)

            box_features = self.model.roi_heads.box_roi_pool(
                features, proposals, images.image_sizes
            )
            box_features = self.model.roi_heads.box_head(box_features)

            class_logits, box_regression = self.model.roi_heads.box_predictor(
                box_features
            )

            detections, detections_indices = self.postprocess_detections(
                class_logits, box_regression, proposals, images.image_sizes
            )

            detections = self.model.transform.postprocess(
                detections, images.image_sizes, original_image_sizes
            )

            embedding_size = self.embedding_output.shape[-1]
            full_embedding_output = self.embedding_output.reshape(
                batch_size, -1, embedding_size
            )
            embedding_output = [
                e[indices]
                for e, indices in zip(full_embedding_output, detections_indices)
            ]
        return detections, embedding_output

    def postprocess_detections(
        # https://github.com/pytorch/vision/blob/8233c9cdf3351e1996249fdb4f3a998f8c9e693d/torchvision/models/detection/roi_heads.py#L668C9-L668C31
        self,
        class_logits,  # type: Tensor
        box_regression,  # type: Tensor
        proposals,  # type: List[Tensor]
        image_shapes,  # type: List[Tuple[int, int]]
    ):
        device = class_logits.device
        num_classes = class_logits.shape[-1]
        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.model.roi_heads.box_coder.decode(box_regression, proposals)
        pred_scores = torch.softmax(class_logits, -1)

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)

        pred_scores_list = pred_scores.split(boxes_per_image, 0)
        all_boxes = []
        all_scores = []
        all_labels = []
        all_keep_indices = []

        for boxes, scores, image_shape in zip(
            pred_boxes_list, pred_scores_list, image_shapes
        ):
            keep_indices = torch.arange(boxes.shape[0], device=device)
            keep_indices = keep_indices.view(-1, 1).expand_as(scores)

            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]
            keep_indices = keep_indices[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)
            keep_indices = keep_indices.reshape(-1)

            # remove low scoring boxes
            inds = torch.where(scores > self.model.roi_heads.score_thresh)[0]
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            keep_indices = keep_indices[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            keep_indices = keep_indices[keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(
                boxes, scores, labels, self.model.roi_heads.nms_thresh
            )

            # keep only topk scoring predictions
            keep = keep[: self.model.roi_heads.detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            keep_indices = keep_indices[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
            all_keep_indices.append(keep_indices)

        results = []
        for i in range(len(all_boxes)):
            results.append(
                {
                    "boxes": all_boxes[i],
                    "labels": all_labels[i],
                    "scores": all_scores[i],
                }
            )

        return results, all_keep_indices

    @property
    def embedding_output(self):
        return self._embedding_output[self.embedding_layer_name]


label_map_model = [
    "__background__",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]
