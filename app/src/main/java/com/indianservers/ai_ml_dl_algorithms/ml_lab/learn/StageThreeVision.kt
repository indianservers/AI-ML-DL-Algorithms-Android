package com.indianservers.ai_ml_dl_algorithms.ml_lab.learn

import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt
import kotlin.math.exp

internal enum class VisionVisualization(val title: String) {
    Classification("Image Classification"), Cnn("CNN"), Detection("Object Detection"),
    Rcnn("R-CNN"), FastRcnn("Fast R-CNN"), FasterRcnn("Faster R-CNN"), Ssd("SSD"),
    Yolo("YOLO"), Retinanet("RetinaNet"), Semantic("Semantic Segmentation"),
    Unet("U-Net"), MaskRcnn("Mask R-CNN"), Instance("Instance Segmentation"),
    Pose("Pose Estimation"), Vit("Vision Transformer"), Embeddings("Image Embeddings"),
    Similarity("Image Similarity");
    companion object { val byTitle = entries.associateBy { it.title } }
}

internal data class VisionBox(val x: Double, val y: Double, val w: Double, val h: Double,
                              val label: String, val confidence: Double)
internal data class SegmentationCell(val label: String, val confidence: Double, val probabilities: List<Double>)

internal object StageThreeVisionEngine {
    /** A small, explicit per-pixel score example; no trained segmentation network is implied. */
    fun segmentationCell(row: Int, col: Int): SegmentationCell {
        require(row in 0..3 && col in 0..3)
        val x = (col + .5) / 4.0
        val y = (row + .5) / 4.0
        fun ellipse(cx: Double, cy: Double, rx: Double, ry: Double): Double =
            (x - cx) * (x - cx) / (rx * rx) + (y - cy) * (y - cy) / (ry * ry)
        val logits = listOf(0.0, 3.5 * (1.0 - ellipse(.34, .46, .24, .32)),
            3.5 * (1.0 - ellipse(.74, .5, .16, .25)))
        val maximum = logits.max()
        val weights = logits.map { exp(it - maximum) }
        val probabilities = weights.map { it / weights.sum() }
        val selected = probabilities.indices.maxBy { probabilities[it] }
        return SegmentationCell(listOf("background", "cat", "dog")[selected],
            probabilities[selected], probabilities)
    }
    val proposals = listOf(
        VisionBox(.12, .18, .45, .55, "cat", .94),
        VisionBox(.15, .2, .44, .52, "cat", .83),
        VisionBox(.64, .28, .24, .45, "dog", .81),
        VisionBox(.6, .25, .28, .49, "dog", .58),
        VisionBox(.32, .5, .18, .22, "cat", .34)
    )

    fun iou(a: VisionBox, b: VisionBox): Double {
        val x1 = max(a.x, b.x); val y1 = max(a.y, b.y)
        val x2 = min(a.x + a.w, b.x + b.w); val y2 = min(a.y + a.h, b.y + b.h)
        val intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        return intersection / (a.w * a.h + b.w * b.h - intersection).coerceAtLeast(1e-9)
    }

    fun nms(boxes: List<VisionBox>, confidence: Double, overlap: Double): List<VisionBox> {
        val kept = mutableListOf<VisionBox>()
        boxes.filter { it.confidence >= confidence }.sortedByDescending { it.confidence }.forEach { box ->
            if (kept.none { it.label == box.label && iou(it, box) > overlap }) kept += box
        }
        return kept
    }

    val imageVectors = mapOf(
        "cat A" to listOf(.93, .08, .11), "cat B" to listOf(.88, .12, .09),
        "dog A" to listOf(.18, .9, .2), "dog B" to listOf(.2, .85, .17),
        "car" to listOf(.12, .17, .95)
    )

    fun cosine(a: List<Double>, b: List<Double>): Double {
        val numerator = a.zip(b).sumOf { it.first * it.second }
        return numerator / (sqrt(a.sumOf { it * it }) * sqrt(b.sumOf { it * it })).coerceAtLeast(1e-9)
    }

    fun ranking(query: String): List<Pair<String, Double>> {
        val q = imageVectors.getValue(query)
        return imageVectors.filterKeys { it != query }.map { it.key to cosine(q, it.value) }.sortedByDescending { it.second }
    }

    fun stages(kind: VisionVisualization): List<String> = when (kind) {
        VisionVisualization.Classification -> listOf("Input image", "Feature extraction", "Class logits", "Softmax probabilities")
        VisionVisualization.Cnn -> listOf("Input matrix", "Sliding 3×3 kernel", "Feature map", "Activation + pooling", "Class head")
        VisionVisualization.Detection -> listOf("Input image", "Candidate boxes", "Class + confidence", "IoU and NMS", "Final boxes")
        VisionVisualization.Rcnn -> listOf("Selective-search proposals", "Crop EACH proposal", "CNN for EACH crop", "Classifier", "Box regression")
        VisionVisualization.FastRcnn -> listOf("Whole image CNN ONCE", "Shared feature map", "External proposals", "ROI pooling", "Class + box heads")
        VisionVisualization.FasterRcnn -> listOf("Shared backbone", "Region Proposal Network", "Objectness + anchor offsets", "ROI pooling", "Detection head")
        VisionVisualization.Ssd -> listOf("Multiple feature-map scales", "Default boxes per cell", "Class + offsets at each scale", "NMS")
        VisionVisualization.Yolo -> listOf("Whole image, one pass", "Dense grid locations", "Simultaneous class + box", "Confidence filter + NMS")
        VisionVisualization.Retinanet -> listOf("Backbone feature pyramid", "Classification subnet", "Box regression subnet", "Focal loss for hard examples")
        VisionVisualization.Semantic -> listOf("Input pixels", "Per-pixel logits", "Softmax class map", "One mask per class")
        VisionVisualization.Unet -> listOf("Downsample encoder", "Bottleneck", "Upsample decoder", "Copy skip features", "Per-pixel mask")
        VisionVisualization.MaskRcnn -> listOf("Faster R-CNN proposals", "ROI Align", "Class + box head", "Parallel instance mask head")
        VisionVisualization.Instance -> listOf("Detect separate objects", "Mask for object A", "Mask for object B", "Instance IDs")
        VisionVisualization.Pose -> listOf("Input person", "Joint heatmaps", "Keypoint confidence", "Connect skeleton")
        VisionVisualization.Vit -> listOf("Split image into patches", "Flatten patch vectors", "Add positional embeddings", "Transformer encoder", "[CLS] head")
        VisionVisualization.Embeddings -> listOf("Images", "Embedding vectors", "Project to 2D", "Neighbor relationships")
        VisionVisualization.Similarity -> listOf("Query image", "Embedding", "Cosine similarity", "Ranked neighbors")
    }
}
