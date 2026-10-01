package com.indianservers.ai_ml_dl_algorithms.ml_lab.learn

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.GlassPanel
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabCyan
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabMuted
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabOrange
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabPurple
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabText
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.SectionTitle
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.SegmentedOption
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.CnnKernelPreset
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.PhaseSixCnnEngines
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.PhaseEightEngines
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.ShapeClass

@Composable
internal fun StageThreeVisionScreen(topic: LearnTopic, kind: VisionVisualization) {
    var selected by remember(topic.id) { mutableIntStateOf(0) }
    var threshold by remember(topic.id) { mutableIntStateOf(5) }
    val stages = StageThreeVisionEngine.stages(kind)
    val detector = kind in setOf(VisionVisualization.Detection, VisionVisualization.Rcnn, VisionVisualization.FastRcnn,
        VisionVisualization.FasterRcnn, VisionVisualization.Ssd, VisionVisualization.Yolo, VisionVisualization.Retinanet,
        VisionVisualization.MaskRcnn)
    val segmentation = kind in setOf(VisionVisualization.Semantic, VisionVisualization.Unet,
        VisionVisualization.Instance, VisionVisualization.MaskRcnn)
    val ranking = kind == VisionVisualization.Similarity || kind == VisionVisualization.Embeddings
    val boxes = remember(threshold) { StageThreeVisionEngine.nms(StageThreeVisionEngine.proposals, threshold / 10.0, .45) }
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        item { SectionTitle(kind.title, when (kind) {
            VisionVisualization.Classification -> "Follow image features into class probabilities"
            VisionVisualization.Cnn -> "Inspect a sliding convolution and its feature map"
            VisionVisualization.Detection -> "Compare candidate boxes, confidence filtering, and suppression"
            VisionVisualization.Rcnn -> "Inspect selective proposals processed by separate CNN crops"
            VisionVisualization.FastRcnn -> "Reuse one feature map across external region proposals"
            VisionVisualization.FasterRcnn -> "Generate proposals with an RPN before region classification"
            VisionVisualization.Ssd -> "Inspect default boxes at several feature map scales"
            VisionVisualization.Yolo -> "See grid locations predict classes and boxes in one pass"
            VisionVisualization.Retinanet -> "Inspect feature pyramids and focal loss for hard examples"
            VisionVisualization.Semantic -> "Select a pixel to inspect its class scores and mask label"
            VisionVisualization.Unet -> "Trace encoder skip features into a per-pixel decoder"
            VisionVisualization.MaskRcnn -> "Inspect detected boxes alongside their instance masks"
            VisionVisualization.Instance -> "See separate object masks and instance identities"
            VisionVisualization.Pose -> "Select a joint to inspect its keypoint confidence"
            VisionVisualization.Vit -> "Follow image patches through position-aware token processing"
            VisionVisualization.Embeddings -> "Compare images projected into an embedding space"
            VisionVisualization.Similarity -> "Rank image neighbors by embedding cosine similarity"
        }) }
        item {
            GlassPanel(Modifier.fillMaxWidth()) {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text("Image workspace", color = LabText, fontWeight = FontWeight.Bold)
                    VisionCanvas(kind, if (detector) boxes else emptyList(), selected) { selected = it }
                    Text(when {
                        segmentation -> {
                            val col=selected%4; val row=selected/4
                            val cell=StageThreeVisionEngine.segmentationCell(row,col)
                            val label=cell.label
                            val detail = if(kind==VisionVisualization.Instance||kind==VisionVisualization.MaskRcnn)
                                "Selected cell ($row,$col): ${if(label=="background")"no instance" else "$label instance"}; toy score %.2f".format(cell.confidence)
                            else "Selected pixel cell ($row,$col): $label; toy score %.2f".format(cell.confidence)
                            if (kind == VisionVisualization.MaskRcnn) "$detail • ${boxes.size} retained boxes" else detail
                        }
                        detector -> "${boxes.size} boxes after confidence filter and class-wise non-maximum suppression."
                        kind == VisionVisualization.Pose -> "Selected joint ${selected % 8 + 1}; joint confidence %.2f".format(.92 - (selected % 8) * .07)
                        kind == VisionVisualization.Vit -> "Selected patch ${selected % 16 + 1}; position is added to its patch embedding."
                        ranking -> "Selected image: ${StageThreeVisionEngine.imageVectors.keys.elementAt(selected % 5)}"
                        else -> "Selected feature location ${selected % 16 + 1}."
                    }, color = LabMuted, fontSize = 12.sp)
                    if (segmentation) Text("Color and confidence come from explicit toy pixel logits and softmax; no segmentation network is trained in this example.",
                        color = LabMuted, fontSize = 11.sp)
                }
            }
        }
        item {
            GlassPanel(Modifier.fillMaxWidth()) {
                Column(verticalArrangement = Arrangement.spacedBy(7.dp)) {
                    Text("Mechanism", color = LabText, fontWeight = FontWeight.Bold)
                    stages.forEachIndexed { i, stage ->
                        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            Text("${i + 1}", color = if (i == selected % stages.size) LabCyan else LabPurple, fontWeight = FontWeight.Bold)
                            Text(stage, color = LabText, fontSize = 12.sp)
                        }
                    }
                    if (kind == VisionVisualization.Rcnn || kind == VisionVisualization.FastRcnn || kind == VisionVisualization.FasterRcnn) {
                        Text(when (kind) {
                            VisionVisualization.Rcnn -> "CNN runs separately for every proposed crop."
                            VisionVisualization.FastRcnn -> "One CNN feature map is shared; ROI pooling reads each proposal."
                            else -> "RPN predicts proposals from the shared feature map before the ROI head."
                        }, color = LabCyan, fontSize = 12.sp)
                    }
                    if (kind == VisionVisualization.Unet) Text("Skip connections copy encoder detail to matching decoder resolutions.", color = LabCyan, fontSize = 12.sp)
                    if (kind == VisionVisualization.Yolo) Text("All grid positions predict in one whole-image pass.", color = LabCyan, fontSize = 12.sp)
                    if (kind == VisionVisualization.Ssd) Text("Default boxes are attached to several feature-map scales.", color = LabCyan, fontSize = 12.sp)
                    if (kind == VisionVisualization.Retinanet) Text("Focal loss downweights easy background examples while the feature pyramid handles scale.", color = LabCyan, fontSize = 12.sp)
                    if(kind==VisionVisualization.Retinanet) {
                        val probability=(.1+selected*.05).coerceIn(.1,.9)
                        val crossEntropy=-kotlin.math.ln(probability)
                        val focal=(1-probability)*(1-probability)*crossEntropy
                        Text("Example pₜ = %.2f: CE = %.3f; focal (γ=2) = %.3f".format(probability,crossEntropy,focal),
                            color=LabOrange,fontSize=12.sp)
                    }
                }
            }
        }
        if (detector) item {
            GlassPanel(Modifier.fillMaxWidth()) {
                Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    Text("Detection geometry", color = LabText, fontWeight = FontWeight.Bold)
                    Text("IoU of two cat proposals = %.3f".format(StageThreeVisionEngine.iou(StageThreeVisionEngine.proposals[0], StageThreeVisionEngine.proposals[1])),
                        color = LabCyan, fontSize = 12.sp)
                    Text("NMS keeps the higher-confidence box when same-class IoU exceeds 0.45.", color = LabMuted, fontSize = 12.sp)
                    Text("Confidence threshold %.1f".format(threshold / 10.0), color = LabText, fontSize = 12.sp)
                    Slider(threshold.toFloat(), { threshold = it.toInt().coerceIn(2, 9) }, valueRange = 2f..9f)
                    boxes.forEach { Text("${it.label}: %.2f at (%.2f, %.2f)".format(it.confidence, it.x, it.y), color = LabMuted, fontSize = 11.sp) }
                }
            }
        }
        if (ranking) item {
            val key = StageThreeVisionEngine.imageVectors.keys.elementAt(selected % 5)
            GlassPanel(Modifier.fillMaxWidth()) {
                Column(verticalArrangement = Arrangement.spacedBy(5.dp)) {
                    Text("Cosine neighbors for $key", color = LabText, fontWeight = FontWeight.Bold)
                    StageThreeVisionEngine.ranking(key).forEach { (label, similarity) ->
                        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                            Text(label, color = LabMuted, fontSize = 12.sp)
                            Text("%.3f".format(similarity), color = LabCyan, fontSize = 12.sp)
                        }
                    }
                    Text("Illustrative fixed embeddings; scores are computed from the shown vectors.", color = LabMuted, fontSize = 11.sp)
                }
            }
        }
        if (kind == VisionVisualization.Cnn) item {
            val image = PhaseSixCnnEngines.presetImage(ShapeClass.Vertical, 7)
            val convolution = PhaseSixCnnEngines.convolve(image, PhaseSixCnnEngines.kernel(CnnKernelPreset.Vertical), 1, 0, selected % 25)
            GlassPanel(Modifier.fillMaxWidth()) {
                Column(verticalArrangement = Arrangement.spacedBy(5.dp)) {
                    Text("Actual 3×3 convolution", color = LabText, fontWeight = FontWeight.Bold)
                    Text("Patch at (${convolution.current.row}, ${convolution.current.col}) → multiply and sum = %.2f".format(convolution.current.sum),
                        color = LabCyan, fontSize = 12.sp)
                    SegmentedOption("Slide kernel", false, Modifier.fillMaxWidth()) { selected++ }
                }
            }
        }
        if(kind==VisionVisualization.Vit) item {
            val image=PhaseSixCnnEngines.presetImage(ShapeClass.X,8)
            val patch=selected.coerceIn(0,15)
            val row=(patch/4)*2;val col=(patch%4)*2
            val pixels=listOf(image[row][col],image[row][col+1],image[row+1][col],image[row+1][col+1])
            val position=PhaseEightEngines.positional(patch+1,4)
            val tokens=listOf("[CLS]")+(0..15).map{"patch$it"}
            val attention=PhaseEightEngines.attention(tokens,dim=4)
            GlassPanel(Modifier.fillMaxWidth()) {
                Column(verticalArrangement=Arrangement.spacedBy(5.dp)) {
                    Text("Patch $patch → token ${patch+1}",color=LabText,fontWeight=FontWeight.Bold)
                    Text("Flattened 2×2 pixels: ${pixels.joinToString { "%.1f".format(it) }}",color=LabMuted,fontSize=12.sp)
                    Text("Positional vector: ${position.joinToString { "%.2f".format(it) }}",color=LabCyan,fontSize=12.sp)
                    Text("Token + position: ${pixels.zip(position).joinToString { "%.2f".format(it.first+it.second) }}",color=LabOrange,fontSize=12.sp)
                    Text("[CLS] attention to selected patch = %.3f".format(attention.cells[0][patch+1].weight),color=LabCyan,fontSize=12.sp)
                }
            }
        }
        if (kind == VisionVisualization.Classification) item {
            val shape=ShapeClass.entries[selected%ShapeClass.entries.size]
            val prediction=PhaseSixCnnEngines.predictShape(PhaseSixCnnEngines.presetImage(shape,7))
            GlassPanel(Modifier.fillMaxWidth()) {
                Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    Text("Tiny offline CNN: ${shape.label} pattern", color = LabText, fontWeight = FontWeight.Bold)
                    ShapeClass.entries.zip(prediction.probabilities).forEach { (label, probability) ->
                        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                            Text(label.label, color = LabMuted, fontSize = 12.sp)
                            Text("${(probability * 100).toInt()}%", color = LabCyan, fontSize = 12.sp)
                        }
                    }
                    Text("Computed from the app's small offline shape-classification model.", color = LabMuted, fontSize = 11.sp)
                    SegmentedOption("Try next shape",false,Modifier.fillMaxWidth()){selected++}
                }
            }
        }
    }
}

@Composable
private fun VisionCanvas(kind: VisionVisualization, boxes: List<VisionBox>, selected: Int, onSelect: (Int) -> Unit) {
    Canvas(Modifier.fillMaxWidth().height(245.dp).background(Color(0xFF081D3A)).pointerInput(kind) {
        detectTapGestures { point -> onSelect(((point.x / size.width * 4).toInt() + (point.y / size.height * 4).toInt() * 4).coerceIn(0, 15)) }
    }) {
        val w = size.width; val h = size.height
        val cyan = LabCyan; val magenta = Color(0xFFFF56C8)
        for (i in 0..4) {
            drawLine(Color(0xFF1F3A62), Offset(i * w / 4, 0f), Offset(i * w / 4, h), 1f)
            drawLine(Color(0xFF1F3A62), Offset(0f, i * h / 4), Offset(w, i * h / 4), 1f)
        }
        when (kind) {
            VisionVisualization.Classification -> {
                val image=PhaseSixCnnEngines.presetImage(ShapeClass.entries[selected%ShapeClass.entries.size],7)
                val cw=w/7; val ch=h/7
                image.forEachIndexed { r,row -> row.forEachIndexed { c,value ->
                    drawRect(cyan.copy(alpha=(.1+.85*value).toFloat()),Offset(c*cw+2,r*ch+2),Size(cw-4,ch-4))
                } }
            }
            VisionVisualization.Semantic, VisionVisualization.Unet, VisionVisualization.Instance, VisionVisualization.MaskRcnn -> {
                for (row in 0..3) for (col in 0..3) {
                    val cell = StageThreeVisionEngine.segmentationCell(row, col)
                    val color = when (cell.label) {
                        "cat" -> cyan
                        "dog" -> magenta
                        else -> Color(0xFF173553)
                    }
                    drawRect(color.copy(alpha = (.18 + .7 * cell.confidence).toFloat()),
                        Offset(col * w / 4 + 2, row * h / 4 + 2), Size(w / 4 - 4, h / 4 - 4))
                    if (selected == row * 4 + col) drawRect(LabOrange,
                        Offset(col * w / 4 + 2, row * h / 4 + 2), Size(w / 4 - 4, h / 4 - 4), style = Stroke(3f))
                }
                if (kind == VisionVisualization.MaskRcnn) boxes.forEach { box ->
                    drawRect(if (box.label == "cat") cyan else magenta,
                        Offset((box.x * w).toFloat(), (box.y * h).toFloat()),
                        Size((box.w * w).toFloat(), (box.h * h).toFloat()), style = Stroke(3f))
                }
                if (kind == VisionVisualization.Unet) {
                    listOf(.08f,.29f,.50f).forEachIndexed { i,y ->
                        drawRect(Color(0xFF12395D),Offset(w*.09f,y*h),Size(w*.18f,h*.13f))
                        drawRect(cyan,Offset(w*.09f,y*h),Size(w*.18f,h*.13f),style=Stroke(3f))
                        drawRect(Color(0xFF12395D),Offset(w*.73f,y*h),Size(w*.18f,h*.13f))
                        drawRect(LabOrange,Offset(w*.73f,y*h),Size(w*.18f,h*.13f),style=Stroke(3f))
                        drawLine(LabPurple,Offset(w*.27f,(y+.065f)*h),Offset(w*.73f,(y+.065f)*h),2f)
                        if(i<2) {
                            drawLine(cyan,Offset(w*.18f,(y+.13f)*h),Offset(w*.18f,(y+.21f)*h),2f)
                            drawLine(LabOrange,Offset(w*.82f,(y+.21f)*h),Offset(w*.82f,(y+.13f)*h),2f)
                        }
                    }
                    drawRect(Color(0xFF12395D),Offset(w*.42f,h*.77f),Size(w*.16f,h*.12f))
                    drawRect(LabPurple,Offset(w*.42f,h*.77f),Size(w*.16f,h*.12f),style=Stroke(3f))
                    drawLine(cyan,Offset(w*.18f,h*.63f),Offset(w*.5f,h*.77f),2f)
                    drawLine(LabOrange,Offset(w*.5f,h*.77f),Offset(w*.82f,h*.63f),2f)
                }
            }
            VisionVisualization.Pose -> {
                val joints = listOf(.5f to .13f, .5f to .31f, .32f to .4f, .68f to .4f,
                    .46f to .58f, .37f to .8f, .54f to .58f, .67f to .8f)
                listOf(0 to 1, 1 to 2, 1 to 3, 1 to 4, 1 to 6, 4 to 5, 6 to 7).forEach { (a,b) ->
                    drawLine(cyan, Offset(joints[a].first * w, joints[a].second * h), Offset(joints[b].first * w, joints[b].second * h), 5f)
                }
                joints.forEachIndexed { i, (x,y) -> drawCircle(if (selected % 8 == i) magenta else LabOrange, 8f, Offset(x*w,y*h)) }
            }
            VisionVisualization.Vit -> {
                for (row in 0..3) for (col in 0..3) {
                    val i = row * 4 + col
                    drawRect(if (selected == i) cyan.copy(alpha = .55f) else LabPurple.copy(alpha = .22f),
                        Offset(col*w/4+3, row*h/4+3), Size(w/4-6,h/4-6))
                }
            }
            VisionVisualization.Embeddings, VisionVisualization.Similarity -> {
                val points = listOf(.18f to .24f, .25f to .3f, .72f to .22f, .76f to .32f, .51f to .78f)
                points.forEachIndexed { i,(x,y) -> drawCircle(if (i == selected % 5) magenta else if (i < 2) cyan else LabOrange,
                    if (i == selected % 5) 15f else 10f, Offset(x*w,y*h)) }
            }
            else -> {
                drawOval(cyan.copy(alpha = .15f), Offset(w*.12f,h*.18f), Size(w*.44f,h*.56f))
                drawOval(magenta.copy(alpha = .15f), Offset(w*.62f,h*.28f), Size(w*.24f,h*.45f))
                boxes.forEach { box -> drawRect(if (box.label == "cat") cyan else magenta,
                    Offset((box.x*w).toFloat(), (box.y*h).toFloat()), Size((box.w*w).toFloat(),(box.h*h).toFloat()),
                    style = Stroke(width = 3f)) }
                if (kind == VisionVisualization.Yolo) for (i in 1..3) {
                    drawLine(LabOrange.copy(alpha = .55f), Offset(i*w/4,0f),Offset(i*w/4,h),2f)
                    drawLine(LabOrange.copy(alpha = .55f), Offset(0f,i*h/4),Offset(w,i*h/4),2f)
                }
                if (kind == VisionVisualization.Rcnn) {
                    // Per-proposal crops leave the image before separate CNN passes.
                    StageThreeVisionEngine.proposals.take(3).forEachIndexed { i, box ->
                        val x=(box.x*w).toFloat(); val y=(box.y*h).toFloat()
                        drawLine(LabOrange,Offset(x,y),Offset(w*.72f+i*13,h*.06f+i*18),2f)
                        drawRect(LabOrange,Offset(w*.72f+i*13,h*.04f+i*18),Size(w*.12f,h*.10f),style=Stroke(2f))
                    }
                }
                if (kind == VisionVisualization.FastRcnn) {
                    // Shared feature map: one coarse grid for all ROI boxes.
                    for (i in 1..5) {
                        drawLine(LabOrange.copy(alpha=.45f),Offset(i*w/6,0f),Offset(i*w/6,h),1.5f)
                        drawLine(LabOrange.copy(alpha=.45f),Offset(0f,i*h/6),Offset(w,i*h/6),1.5f)
                    }
                }
                if (kind == VisionVisualization.FasterRcnn) {
                    // RPN anchor centers live on the backbone feature map.
                    for (row in 1..3) for (col in 1..4) {
                        drawCircle(LabOrange,3f,Offset(col*w/5,row*h/4))
                    }
                    drawRect(LabOrange,Offset(w*.22f,h*.2f),Size(w*.3f,h*.4f),style=Stroke(2f))
                }
                if (kind == VisionVisualization.Ssd) {
                    // Different default-box scales are attached to feature-map locations.
                    listOf(.11f,.19f,.28f).forEachIndexed { i,r ->
                        drawRect(listOf(LabOrange,LabPurple,LabCyan)[i],Offset(w*.48f-r*w,h*.5f-r*h),
                            Size(2*r*w,2*r*h),style=Stroke(2f))
                    }
                }
                if (kind == VisionVisualization.Retinanet) {
                    // Pyramid bands feed two parallel prediction heads.
                    for (i in 0..2) {
                        drawRect(LabOrange.copy(alpha=.18f+i*.11f),Offset(w*(.05f+i*.06f),h*(.05f+i*.07f)),
                            Size(w*(.45f-i*.08f),h*(.17f-i*.02f)))
                    }
                    drawLine(LabCyan,Offset(w*.5f,h*.1f),Offset(w*.8f,h*.18f),2f)
                    drawLine(magenta,Offset(w*.5f,h*.1f),Offset(w*.8f,h*.3f),2f)
                }
            }
        }
    }
}
