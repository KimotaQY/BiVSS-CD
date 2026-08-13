from PIL import Image

from bivss_cd.video import pseudo_video


def test_verified_pseudo_video_uses_one_based_default_jpegs(tmp_path):
    image_t1 = tmp_path / "t1.png"
    image_t2 = tmp_path / "t2.png"
    Image.new("RGB", (8, 8), "red").save(image_t1)
    Image.new("RGB", (8, 8), "blue").save(image_t2)

    with pseudo_video(image_t1, image_t2) as root:
        assert sorted(path.name for path in root.iterdir()) == ["1.jpg", "2.jpg"]
