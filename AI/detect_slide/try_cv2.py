
import cv2
import numpy as np

class CropSlide:
    def __init__(self, image):
        pass
def order_points(pts):
    #@ Function này nhận vào 4 điểm bất kỳ của một tứ giác dưới dạng mảng `pts`.
    #@ Mục tiêu là chuẩn hóa thứ tự điểm để các bước xử lý sau luôn dùng cùng một quy ước:
    #@ top-left, top-right, bottom-right, bottom-left.
    #@ Ý tưởng hoạt động:
    #@ - Tính tổng `x + y` cho từng điểm.
    #@ - Điểm có tổng nhỏ nhất thường nằm ở góc trên-trái.
    #@ - Điểm có tổng lớn nhất thường nằm ở góc dưới-phải.
    #@ - Tính hiệu `y - x` để phân biệt hai điểm còn lại.
    #@ - Điểm có hiệu nhỏ nhất thường là góc trên-phải.
    #@ - Điểm có hiệu lớn nhất thường là góc dưới-trái.
    #@ Kết quả trả về là mảng `rect` kích thước (4, 2), rất quan trọng vì các function khác
    #@ giả định thứ tự điểm luôn cố định khi kiểm tra góc, vẽ polygon, hoặc in nhãn TL/TR/BR/BL.
    """Sắp xếp 4 điểm: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left: x+y nhỏ nhất
    rect[2] = pts[np.argmax(s)]   # bottom-right: x+y lớn nhất
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right: y-x nhỏ nhất
    rect[3] = pts[np.argmax(diff)]  # bottom-left: y-x lớn nhất
    return rect


def resize_for_display(img, max_width=1280, max_height=720):
    #@ Function này chỉ phục vụ việc hiển thị ảnh bằng `imshow`, không thay đổi dữ liệu detect thật.
    #@ Nó đọc chiều cao và chiều rộng gốc của ảnh, sau đó tính một hệ số scale sao cho:
    #@ - Ảnh không vượt quá `max_width`
    #@ - Ảnh không vượt quá `max_height`
    #@ - Ảnh không bị phóng to thêm nếu vốn đã nhỏ hơn khung hiển thị
    #@ Sau đó function resize ảnh theo đúng tỉ lệ gốc để tránh méo hình.
    #@ Điều này giải quyết lỗi chỉ nhìn thấy một góc ảnh khi ảnh gốc quá lớn so với màn hình.
    height, width = img.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)


def is_valid_slide_quad(approx, image_area, img_h, img_w):
    #@ Function này là bộ lọc chất lượng cho một contour đã được xấp xỉ bằng `approxPolyDP`.
    #@ Nhiệm vụ của nó là quyết định xem contour đó có thật sự giống vùng slide/màn chiếu hay không.
    #@ Cách hoạt động theo từng lớp kiểm tra:
    #@ - Filter 1: phải đúng 4 đỉnh, vì slide được giả định là tứ giác.
    #@ - Filter 2: diện tích phải đủ lớn để tránh nhiễu nhỏ, nhưng cũng không được gần kín toàn ảnh.
    #@ - Filter 3: contour phải lồi, vì hình slide thông thường không bị lõm.
    #@ - Filter 4: tỉ lệ cạnh dài/cạnh ngắn phải nằm trong khoảng hợp lý cho slide.
    #@ - Filter 5: ghi chú rằng một số vật thể sáng khác có thể đã bị loại từ bước diện tích.
    #@ - Filter 6: tính góc tại từng đỉnh để loại các hình quá méo, quá nhọn, hoặc quá tù.
    #@ Nếu qua được toàn bộ điều kiện, function trả về `(True, "ok")`.
    #@ Nếu trượt một điều kiện, function trả về `(False, reason)` để debug lý do bị loại.
    """
    Kiểm tra một quadrilateral có phải là slide hợp lệ không.
    Trả về (is_valid, reason) để dễ debug.
    """
    # --- Filter 1: Phải đúng 4 điểm ---
    if len(approx) != 4:
        return False, f"not quad ({len(approx)} points)"

    pts = approx.reshape(4, 2).astype("float32")

    # --- Filter 2: Diện tích tối thiểu ---
    area = cv2.contourArea(approx)
    area_ratio = area / image_area
    if area_ratio < 0.10:
        return False, f"too small ({area_ratio:.2%} of image)"
    if area_ratio > 0.98:
        return False, f"too large ({area_ratio:.2%} of image)"

    # --- Filter 3: Convex (slide là hình lồi) ---
    if not cv2.isContourConvex(approx):
        return False, "not convex"

    # --- Filter 4: Aspect ratio hợp lệ ---
    # Dùng minAreaRect để lấy kích thước thực, không bị ảnh hưởng bởi góc nghiêng.
    rect = cv2.minAreaRect(approx)
    rw, rh = rect[1]
    if rw == 0 or rh == 0:
        return False, "zero dimension"
    aspect = max(rw, rh) / min(rw, rh)
    # Slide 4:3 = 1.33, 16:9 = 1.78. Cho phép rộng hơn do góc chụp lệch.
    if not (1.05 <= aspect <= 2.5):
        return False, f"bad aspect ratio ({aspect:.2f})"

    # --- Filter 5: Không phải hình vuông quá hoàn hảo ---
    # Một số fixture sáng trong phòng có thể tạo contour đánh lừa, nhưng phần lớn
    # đã bị loại ở bước diện tích nên chỉ giữ ghi chú này để nhắc logic thiết kế.

    # --- Filter 6: Các góc phải đủ "vuông" ---
    # Tính góc tại mỗi đỉnh, slide thực không nên có góc < 45° hay > 135°.
    ordered = order_points(pts)
    for i in range(4):
        p0 = ordered[i]
        p1 = ordered[(i + 1) % 4]
        p2 = ordered[(i + 2) % 4]
        v1 = p0 - p1
        v2 = p2 - p1
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        if not (45 <= angle <= 135):
            return False, f"corner angle out of range ({angle:.1f}° at vertex {i})"

    return True, "ok"


def detect_edges_multi_method(grey_img):
    #@ Function này tạo ra nhiều phiên bản edge map khác nhau từ cùng một ảnh grayscale.
    #@ Ý tưởng là không phụ thuộc vào một ngưỡng duy nhất, vì điều kiện ánh sáng của ảnh chụp slide
    #@ có thể thay đổi mạnh giữa các ảnh.
    #@ Các bước:
    #@ - Blur ảnh trước để giảm nhiễu và làm cạnh ổn định hơn.
    #@ - Method 1: dùng Canny với threshold tự tính từ median của ảnh đã blur.
    #@ - Method 2: tăng tương phản cục bộ bằng CLAHE rồi mới chạy Canny.
    #@ - Method 3: adaptive threshold rồi chạy Canny trên kết quả đó.
    #@ Function trả về danh sách `(label, edge_map)` để bước sau thử từng phương pháp và chọn kết quả tốt nhất.
    """
    Thử nhiều cách detect edge, trả về list các edge map.
    Thay vì hardcode threshold 200, dùng adaptive + Canny.
    """
    results = []
    blurred = cv2.GaussianBlur(grey_img, (5, 5), 0)

    # Method 1: Canny với auto threshold (dùng median)
    median = np.median(blurred)
    low = int(max(0, 0.5 * median))
    high = int(min(255, 1.5 * median))
    edges_canny = cv2.Canny(blurred, low, high)
    results.append(("canny_auto", edges_canny))

    # Method 2: Canny sau khi tăng contrast (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    edges_clahe = cv2.Canny(enhanced, 30, 100)
    results.append(("canny_clahe", edges_clahe))

    # Method 3: Adaptive threshold -> Canny
    adaptive = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    edges_adaptive = cv2.Canny(adaptive, 10, 50)
    results.append(("adaptive", edges_adaptive))

    return results


def find_best_quad(edge_map, image_area, img_h, img_w, debug_label=""):
    #@ Function này nhận vào một edge map và cố gắng tìm ra tứ giác phù hợp nhất đại diện cho slide.
    #@ Trình tự hoạt động:
    #@ - Dùng `dilate` để nối các đoạn cạnh bị đứt, giúp contour khép kín tốt hơn.
    #@ - Tìm contour ngoài cùng và sắp xếp theo diện tích giảm dần.
    #@ - Chỉ xét một số contour lớn nhất vì slide thường là vùng nổi bật trong ảnh.
    #@ - Với mỗi contour, thử nhiều giá trị `epsilon` cho `approxPolyDP`.
    #@ - Mỗi contour sau khi xấp xỉ sẽ được đưa qua `is_valid_slide_quad(...)`.
    #@ - Nếu hợp lệ, function sắp xếp thứ tự điểm bằng `order_points(...)` rồi trả về ngay.
    #@ Nếu không có contour nào đạt yêu cầu, function trả về `(None, 0)`.
    #@ Function cũng in lý do contour lớn nhất bị loại để phục vụ debug.
    """
    Tìm quadrilateral tốt nhất từ một edge map.
    Trả về (corners, area) hoặc (None, 0).
    """
    # Dilate để nối cạnh bị đứt
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edge_map, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for i, cnt in enumerate(contours[:8]):  # Chỉ xét 8 contour lớn nhất
        area = cv2.contourArea(cnt)
        if area < image_area * 0.08:
            break  # Đã sort, các contour sau đều nhỏ hơn

        peri = cv2.arcLength(cnt, True) #? perimeter of the contour (True means we assume the contour is closed)

        # Thử nhiều epsilon để approxPolyDP
        for eps_factor in [0.01, 0.02, 0.03, 0.05]:
            approx = cv2.approxPolyDP(cnt, eps_factor * peri, True) #? approxPolyDP simplify contour hiện tại,
                                #? với sai số cho phép là eps * peri, tức là chi vi càng lớn thì sai số sẽ càng lớn
            valid, reason = is_valid_slide_quad(approx, image_area, img_h, img_w)
            if valid:
                pts = approx.reshape(4, 2).astype("float32")
                ordered = order_points(pts)
                print(f"  [{debug_label}] Found valid quad at contour #{i}, eps={eps_factor}")
                return ordered, area

    return None, 0

def fallback_corners(img_h, img_w, margin=0.10):
    #@ Đây là phương án dự phòng khi toàn bộ quá trình detect thất bại.
    #@ Function không cố đoán slide nữa mà tạo một hình chữ nhật "an toàn" ở giữa ảnh.
    #@ `margin=0.10` nghĩa là chừa ra 10% mép ảnh ở mỗi phía, nên hình chữ nhật trung tâm
    #@ sẽ chiếm khoảng 80% chiều rộng và 80% chiều cao.
    #@ Thứ tự điểm trả về vẫn theo chuẩn TL, TR, BR, BL để đồng nhất với toàn bộ chương trình.
    #@ Ý nghĩa của fallback là đảm bảo chương trình luôn có một bộ 4 góc hợp lệ để tiếp tục workflow.
    """
    Fallback: hình chữ nhật chiếm 80% trung tâm ảnh.
    User sẽ kéo chỉnh từ đây trong UI.
    """
    mx = int(img_w * margin)
    my = int(img_h * margin)
    return np.array([
        [mx, my],  # top-left
        [img_w - mx, my],  # top-right
        [img_w - mx, img_h - my],  # bottom-right
        [mx, img_h - my],  # bottom-left
    ], dtype="float32")


def detect_screen_corners(image_path, debug=True):
    #@ Đây là function chính của chương trình, chịu trách nhiệm đọc ảnh, detect slide,
    #@ hiển thị kết quả và trả về 4 góc cuối cùng.
    #@ Luồng hoạt động tổng quát:
    #@ - Đọc ảnh từ `image_path`. Nếu không đọc được thì báo lỗi ngay.
    #@ - Tạo bản sao `orig` để vẽ kết quả hiển thị mà không làm hỏng ảnh đầu vào.
    #@ - Lấy kích thước ảnh và tính diện tích ảnh để phục vụ các bộ lọc phía sau.
    #@ - Chuyển ảnh sang grayscale vì các bước detect cạnh không cần thông tin màu.
    #@ - Gọi `detect_edges_multi_method(...)` để sinh ra nhiều edge map khác nhau.
    #@ - Duyệt từng edge map, với mỗi edge map gọi `find_best_quad(...)` để tìm tứ giác tốt nhất.
    #@ - So sánh theo diện tích contour hợp lệ và giữ lại kết quả tốt nhất.
    #@ - Nếu không method nào detect được, dùng `fallback_corners(...)` để tạo 4 góc mặc định.
    #@ Sau khi có `best_corners`, function sẽ vẽ polygon, đánh dấu từng góc, ghi nhãn trạng thái,
    #@ resize ảnh để vừa màn hình rồi hiển thị bằng `imshow`.
    #@ Function trả về:
    #@ - `best_corners`: 4 góc cuối cùng
    #@ - `is_detected`: cho biết đây là detect thật hay fallback
    """
    Detect 4 góc của slide trong ảnh chụp.

    Returns:
        corners: np.array shape (4, 2) theo thứ tự TL, TR, BR, BL
        is_detected: True nếu detect được, False nếu dùng fallback
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {image_path}")

    orig = img.copy()
    img_h, img_w = img.shape[:2]
    image_area = img_h * img_w

    #@ chuyển ảnh về đen trắng
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Thử từng method detect edge -> return 3 tuples (method_name, array) after applying canny
    edge_methods = detect_edges_multi_method(grey_img)

    best_corners = None
    best_area = 0

    for label, edge_map in edge_methods:
        if debug:
            print(f"Trying method: {label}")
        corners, area = find_best_quad(edge_map, image_area, img_h, img_w, debug_label=label)
        if corners is not None and area > best_area:
            best_corners = corners
            best_area = area

    is_detected = best_corners is not None
    if not is_detected:
        print("Không detect được slide -> dùng fallback (80% trung tâm ảnh)")
        best_corners = fallback_corners(img_h, img_w)
    return best_corners, is_detected

def visualize_result(img_path, best_corners, is_detected):
    img = cv2.imread(img_path)
    orig = img.copy()

    # --- Visualize kết quả ---
    color = (0, 255, 0) if is_detected else (0, 165, 255)  # xanh nếu detect, cam nếu fallback
    label_text = "DETECTED" if is_detected else "FALLBACK"

    pts = best_corners.astype(np.int32)

    # Vẽ polygon
    cv2.polylines(orig, [pts], isClosed=True, color=color, thickness=3)

    # Vẽ từng góc và label
    labels = ["TL", "TR", "BR", "BL"]
    for i, (x, y) in enumerate(pts):
        cv2.circle(orig, (x, y), 12, (0, 0, 255), -1)
        cv2.putText(orig, labels[i], (x + 15, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.putText(orig, label_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    display_img = resize_for_display(orig)
    cv2.imshow("Slide Corner Detection", display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# --- Test ---
if __name__ == "__main__":
    #@ Khối này chỉ chạy khi file được chạy trực tiếp bằng Python, ví dụ `python test.py`.
    #@ Nếu file chỉ được import như một module từ file khác thì phần bên dưới sẽ không chạy.
    #@ Cách chương trình hoạt động trong khối main:
    #@ - Import `sys` để đọc đối số dòng lệnh.
    #@ - Gán sẵn một `image_path` mặc định để tiện test nhanh.
    #@ - Nếu người dùng truyền đường dẫn ảnh ở command line, chương trình sẽ dùng đường dẫn đó.
    #@ - Gọi `detect_screen_corners(image_path, debug=True)` để chạy toàn bộ pipeline detect.
    #@ - Bên trong pipeline, chương trình sẽ:
    #@   đọc ảnh -> chuyển grayscale -> tạo nhiều edge map -> tìm contour/tứ giác tốt nhất
    #@   -> kiểm tra tính hợp lệ -> nếu cần thì fallback -> vẽ kết quả -> resize để hiển thị.
    #@ - Sau cùng, chương trình in ra trạng thái `Detected` hoặc `Fallback`
    #@   và in tọa độ 4 góc theo thứ tự TL, TR, BR, BL.
    import sys
    import numpy as np

    image_path = r"data/Messenger_creation_35476E35-E07E-4B81-A1FC-C02714A07C53.jpeg"
    if len(sys.argv) > 1:
        image_path = sys.argv[1]

    corners, detected = detect_screen_corners(image_path, debug=True)


    print(f"\nResult: {'Detected' if detected else 'Fallback'}")
    print(f"Corners (TL, TR, BR, BL):\n{corners}")

# accuracy: 
#     17 ảnh detech đúng
#     7 ảnh detech sai
#     13 còn lại là fall back
