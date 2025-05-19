import numpy as np
from mean_average_precision import MetricBuilder


def xywh_to_xyxy(boxes_list):
    """
    Convert boxes from [x_center, y_center, w, h] to [x1, y1, x2, y2] format (both relative coordinates)
    """
    xyxy_boxes = []
    for boxes in boxes_list:
        if sum(len(b) for b in boxes) == 0:
            xyxy_boxes.append(np.array([]))
            continue
        new_boxes = []
        for box in boxes:
            # Convert relative coordinates to absolute coordinates
            x_center, y_center, width, height = box
            x1 = (x_center - width / 2)
            y1 = (y_center - height / 2)
            x2 = (x_center + width / 2)
            y2 = (y_center + height / 2)
            new_boxes.append(np.array([x1, y1, x2, y2]))
        xyxy_boxes.append(np.array(new_boxes))
    return xyxy_boxes

def calculate_map_from_boxes(pred_boxes_list, true_boxes_list, conf_scores_list, img_size=(640, 640)):
    """
    Calculate mAP metrics from predicted and true boxes (boxes are xyxy relative coordinates)

    Returns a dictionary containing mAP50 and mAP50-95 values
    """
    # print list of available metrics
    print(MetricBuilder.get_metrics_list())
    metric_fn = MetricBuilder.build_evaluation_metric(
        "map_2d",           # Use 2D bounding box metric
        async_mode=False,   # Calculate synchronously 
        num_classes=1
    )
    
    for pred_boxes, true_boxes, conf_scores in zip(pred_boxes_list, true_boxes_list, conf_scores_list):
        # Format predictions for library
        preds = []
        tbs = []
        if len(pred_boxes) == 0 and len(true_boxes) == 0:
            continue
        
        # [xmin, ymin, xmax, ymax, class_id, confidence]
        for i, box in enumerate(pred_boxes):
            if len(box) == 0:
                continue
            x1, y1, x2, y2 = box
            preds.append(np.array([
                x1 * img_size[0], 
                y1 * img_size[1], 
                x2 * img_size[0], 
                y2 * img_size[1], 
                0, 
                conf_scores[i]
            ]))

        # [xmin, ymin, xmax, ymax, class_id, difficulty, crowd]
        # difficult=0 means the object is not difficult to detect
        # crowd=0 means the object is not in a crowd
        for box in true_boxes:
            if len(box) == 0:
                continue
            x1, y1, x2, y2 = box
            tbs.append(np.array([
                x1 * img_size[0], 
                y1 * img_size[1], 
                x2 * img_size[0], 
                y2 * img_size[1], 
                0,
                0,
                0
            ])) 

        # Add detections for this image
        metric_fn.add(np.array(preds), np.array(tbs))

    # Calculate metrics across standard IoU thresholds
    results = metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05))
    return {"mAP50": results[0.5][0]['ap'], "mAP50-95": results["mAP"]}
