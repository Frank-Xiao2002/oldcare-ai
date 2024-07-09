import cv2
import numpy as np
from ultralytics import YOLO


def is_point_inside_polygon(point, polygon):
    """
    Determine if a point is inside a given polygon or not
    Polygon is a list of (x,y) pairs.
    :param point: a tuple of (x, y) coordinates
    :param polygon: a list of (x, y) pairs
    """
    x, y = point
    odd_intersects = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if y1 < y2:
            ymin = y1
            ymax = y2
        else:
            ymin = y2
            ymax = y1
        if y == ymax:
            y = y + 0.0001
        if ymin < y <= ymax:
            if x1 > x2:
                xmin = x2
            else:
                xmin = x1
            if x <= xmin:
                odd_intersects = not odd_intersects
            else:
                if abs(x1 - x2) > 0:
                    m_edge = (y2 - y1) / (x2 - x1)
                else:
                    m_edge = 999999999
                if abs(x1 - x) > 0:
                    m_point = (y - y1) / (x - x1)
                else:
                    m_point = 999999999
                if m_point >= m_edge:
                    odd_intersects = not odd_intersects
    return odd_intersects


def get_people_center(results):
    """
    get a list of people center coordinates
    :param results: the results from the YOLO model
    :return: a list of people center coordinates
    """
    # Extract bounding boxes and class IDs
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Convert to numpy array for easier manipulation
    class_ids = results[0].boxes.cls.cpu().numpy()  # Assuming class IDs are stored here

    # Filter for "person" class ID and calculate center coordinates
    person_centers = []
    for box, class_id in zip(boxes, class_ids):
        if class_id == 0:  # Check if the class ID corresponds to "person"
            x_min, y_min, x_max, y_max = box[:4]  # Extract bounding box coordinates
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            person_centers.append((center_x, center_y))
    return person_centers


def are_people_inside_area(image, polygons, save_name='processed.jpg'):
    """
    Check if people are inside the given areas
    :param image:  the image to be processed
    :param polygons: list of polygons, each polygon is a list of (x, y) pairs
    :param save_name: the name of the image to be saved, default is 'processed.jpg'
    :return: true if people are inside the given areas, false otherwise
    """
    model = YOLO('yolov8n.pt')
    results = model.predict(source=image)
    centers = get_people_center(results)
    for center in centers:
        for polygon in polygons:
            if is_point_inside_polygon(center, polygon):
                depict_intrusion(image, polygons, results, save_name)
                return True
    return False


def depict_intrusion(image, polygons, results, save_img_name='image_with_boxes.jpg'):
    """
    Draw the polygons and bounding boxes of people on the image
    :param image: image's path
    :param polygons: list of polygons, each polygon is a list of (x, y) pairs
    :param results: the results from the YOLO model
    :param save_img_name: the name of the image to be saved, default is 'image_with_boxes.jpg'
    """
    cv2_image = cv2.imread(image)
    for polygon in polygons:
        # Ensure the polygon points are in the correct shape for cv2.polylines
        pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
        # Draw the polygon outline
        cv2.polylines(cv2_image, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
    # Filter and draw bounding boxes for "person" class
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes
    class_ids = results[0].boxes.cls.cpu().numpy()  # Class IDs
    for box, class_id in zip(boxes, class_ids):
        if class_id == 0:  # Check if the class ID corresponds to "person"
            x_min, y_min, x_max, y_max = box[:4]
            # convert x_min to integer
            x_min = int(x_min)
            y_min = int(y_min)
            x_max = int(x_max)
            y_max = int(y_max)
            cv2.rectangle(cv2_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Use green color for humans
    # Save or display the modified image
    cv2.imwrite(save_img_name, cv2_image)  # Update the path as needed
    print(f'Intrusion detected! Image saved as "{save_img_name}"')
    # cv2.imshow('Image with Human Boxes', cv2_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
