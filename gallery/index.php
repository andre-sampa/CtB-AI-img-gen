<?php
// Function to get supported image files
function getImages($directory) {
    $allowedTypes = array('jpg', 'jpeg', 'png', 'gif');
    $files = array();
    
    if (is_dir($directory)) {
        $dirContent = scandir($directory);
        foreach ($dirContent as $file) {
            $extension = strtolower(pathinfo($file, PATHINFO_EXTENSION));
            if (in_array($extension, $allowedTypes)) {
                $files[] = $file;
            }
        }
        // Sort files in reverse order
        rsort($files);
    }
    return $files;
}
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dark Theme Image Gallery</title>
    <style>
        body {
            background-color: #1a1a1a;
            color: #ffffff;
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
        }

        .gallery-container {
            max-width: 1200px;
            margin: 0 auto;
        }

        h1 {
            text-align: center;
            color: #ffffff;
            margin-bottom: 30px;
        }

        .gallery {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 20px;
            padding: 20px;
        }

        .gallery-item {
            position: relative;
            overflow: hidden;
            border-radius: 8px;
            background-color: #2d2d2d;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            transition: transform 0.3s ease;
            cursor: pointer;
        }

        .gallery-item:hover {
            transform: scale(1.02);
        }

        .gallery-item img {
            width: 100%;
            height: 200px;
            object-fit: cover;
            display: block;
        }

        .image-caption {
            padding: 10px;
            background-color: rgba(0, 0, 0, 0.7);
            color: #ffffff;
            font-size: 0.9em;
            text-align: center;
        }

        /* Lightbox styles */
        .lightbox {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.9);
            z-index: 1000;
            justify-content: center;
            align-items: center;
            cursor: pointer;
        }

        .lightbox.active {
            display: flex;
        }

        .lightbox img {
            max-width: 90%;
            max-height: 90vh;
            object-fit: contain;
            border: 2px solid #ffffff;
            box-shadow: 0 0 20px rgba(0, 0, 0, 0.5);
        }

        .close-button {
            position: fixed;
            top: 20px;
            right: 20px;
            color: #ffffff;
            font-size: 30px;
            cursor: pointer;
            background: none;
            border: none;
            padding: 10px;
            z-index: 1001;
        }

        @media (max-width: 768px) {
            .gallery {
                grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            }
        }
    </style>
</head>
<body>
    <div class="gallery-container">
    <h1>========== CtB-AI-img-gen - Image Gallery ==========</h1>
    <div class="gallery">
            <?php
            // Specify your images directory
            $imageDirectory = 'images/';
            $images = getImages($imageDirectory);

            foreach ($images as $image) {
                echo '<div class="gallery-item" onclick="openLightbox(\'' . $imageDirectory . $image . '\')">';
                echo '<img src="' . $imageDirectory . $image . '" alt="' . pathinfo($image, PATHINFO_FILENAME) . '">';
                echo '<div class="image-caption">' . pathinfo($image, PATHINFO_FILENAME) . '</div>';
                echo '</div>';
            }

            if (empty($images)) {
                echo '<p style="text-align: center; grid-column: 1/-1;">No images found in the gallery.</p>';
            }
            ?>
        </div>
    </div>

    <!-- Lightbox container -->
    <div class="lightbox" id="lightbox" onclick="closeLightbox()">
        <button class="close-button" onclick="closeLightbox()">&times;</button>
        <img id="lightbox-img" src="" alt="Lightbox image">
    </div>

    <script>
        function openLightbox(imageSrc) {
            const lightbox = document.getElementById('lightbox');
            const lightboxImg = document.getElementById('lightbox-img');
            lightboxImg.src = imageSrc;
            lightbox.classList.add('active');
            // Prevent scrolling when lightbox is open
            document.body.style.overflow = 'hidden';
        }

        function closeLightbox() {
            const lightbox = document.getElementById('lightbox');
            lightbox.classList.remove('active');
            // Restore scrolling
            document.body.style.overflow = 'auto';
        }

        // Close lightbox with Escape key
        document.addEventListener('keydown', function(event) {
            if (event.key === 'Escape') {
                closeLightbox();
            }
        });

        // Prevent lightbox from closing when clicking on the image
        document.getElementById('lightbox-img').addEventListener('click', function(event) {
            event.stopPropagation();
        });
    </script>
</body>
</html>