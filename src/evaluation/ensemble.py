import numpy as np
from ensemble_boxes import weighted_boxes_fusion

def ensemble_predict_wbf(models, image_files, iou_threshold=0.5, confidence_threshold=0.25, skip_box_thr=0.0001):
    """
    Performs ensemble prediction using Weighted Box Fusion on multiple YOLO models.
    
    Returns class predictions (0=anomaly detected, 1=no anomaly) and corresponding boxes
    """
    # Initialize results list
    ensemble_results = []
    boxes_list = []
    conf_list = []
    
    # Process each image
    for img_path in image_files:
        # Get predictions from each model
        all_boxes = []
        all_scores = []
        all_classes = []
        
        for model in models:
            results = model.predict(img_path, conf=confidence_threshold, verbose=False)[0]
            
            if len(results.boxes) > 0:
                # Extract boxes, scores and classes
                # (boxes in relative xyxy format)
                all_boxes.append(results.boxes.xyxy.cpu().numpy() / np.array([results.orig_shape[1], results.orig_shape[0], results.orig_shape[1], results.orig_shape[0]]))
                all_scores.append(results.boxes.conf.cpu().numpy())
                all_classes.append(results.boxes.cls.cpu().numpy())
        
        # If no detections from any model
        if sum(len(b) for b in all_boxes) == 0:
            boxes_list.append(np.array([]))
            conf_list.append(np.array([]))
            ensemble_results.append(1)  # 1 (no anomaly)
            continue
            
        # Apply Weighted Box Fusion
        fused_boxes, fused_scores, fused_classes = weighted_box_fusion(
            all_boxes, all_scores, all_classes, iou_threshold, skip_box_thr
        )
        
        boxes_list.append(fused_boxes)
        conf_list.append(fused_scores)
        # Determine classification based on fused results
        if len(fused_boxes) > 0:
            ensemble_results.append(0)  # 0 (anomaly detected)
        else:
            ensemble_results.append(1)  # 1 (no anomaly)
    
    return np.array(ensemble_results), boxes_list, conf_list

def weighted_box_fusion(boxes_list, scores_list, classes_list, iou_thr=0.5, skip_box_thr=0.0001):
    """
    Perform Weighted Box Fusion using the ensemble-boxes package.
    
    Returns (fused_boxes, fused_scores, fused_classes)
    """
    if not boxes_list or all(len(b) == 0 for b in boxes_list): 
        return np.array([]), np.array([]), np.array([])

    # Convert lists into expected format for WBF library
    boxes_list = [np.array(b).tolist() for b in boxes_list]
    scores_list = [np.array(s).tolist() for s in scores_list]
    classes_list = [np.array(c).tolist() for c in classes_list]
    
    # Apply Weighted Box Fusion
    fused_boxes, fused_scores, fused_classes = weighted_boxes_fusion(
        boxes_list, scores_list, classes_list, iou_thr=iou_thr, skip_box_thr=skip_box_thr
    )
    
    return np.array(fused_boxes), np.array(fused_scores), np.array(fused_classes)