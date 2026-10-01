package com.indianservers.ai_ml_dl_algorithms.ml_lab.learn

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
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
import kotlin.math.pow

internal data class CnnArchitectureFrame(val stages: List<String>, val numbers: List<Pair<String, Double>>,
                                         val equation: String, val distinction: String)

internal object StageThreeCnnArchitectureEngine {
    fun frame(kind: DeepVisualization, control: Int): CnnArchitectureFrame = when (kind) {
        DeepVisualization.Lenet -> CnnArchitectureFrame(
            listOf("28×28×1 input", "5×5 Conv → 24×24×6", "2×2 Pool → 12×12×6", "5×5 Conv → 8×8×16", "Pool", "FC → 10"),
            listOf("First conv parameters" to (5*5*1*6+6).toDouble(), "Second conv parameters" to (5*5*6*16+16).toDouble()),
            "output width = (input − kernel)/stride + 1", "Small grayscale digit network with alternating convolution and pooling.")
        DeepVisualization.Alexnet -> CnnArchitectureFrame(
            listOf("227×227 RGB", "11×11 stride-4 Conv", "ReLU + max pool", "5×5 Conv", "3×3 Conv stack", "FC layers"),
            listOf("Early receptive field" to 11.0, "First conv weights" to (11*11*3*96).toDouble()),
            "Conv1 parameters = 11×11×3×96 + 96", "Large early kernel and deep ReLU convolution stack distinguish AlexNet.")
        DeepVisualization.Vgg -> {
            val blocks = if (control % 2 == 0) listOf(2,2,3,3,3) else listOf(2,2,4,4,4)
            CnnArchitectureFrame(blocks.mapIndexed { i,n -> "Block ${i+1}: $n × 3×3 Conv → Pool" },
                listOf("Convolution layers" to blocks.sum().toDouble(), "Effective field of 2 stacked 3×3" to 5.0),
                "two 3×3 layers: 18C² weights vs one 5×5: 25C²", "Repeated small kernels build depth with fewer weights per receptive field.")
        }
        DeepVisualization.Inception -> CnnArchitectureFrame(
            listOf("Input", "1×1 branch", "1×1 → 3×3 branch", "1×1 → 5×5 branch", "Pool → 1×1 branch", "Concatenate"),
            listOf("Branch channels" to 16.0, "Concatenated channels" to 64.0),
            "output channels = sum(branch output channels)", "Parallel receptive-field sizes inspect the same input, then concatenate features.")
        DeepVisualization.Resnet -> {
            val x = control/10.0; val residual = .4*x+.2
            CnnArchitectureFrame(listOf("Input x", "Conv → ReLU → Conv = F(x)", "Skip x", "Elementwise ADD", "ReLU output"),
                listOf("x" to x, "F(x)" to residual, "F(x)+x" to x+residual),
                "output = ReLU(F(x) + x)", "The identity skip is ADDED to the transformed branch.")
        }
        DeepVisualization.Densenet -> {
            val growth = control.coerceIn(2,9)
            CnnArchitectureFrame(listOf("x₀: 8 channels", "H₁([x₀]): +$growth", "H₂([x₀,x₁]): +$growth", "H₃([x₀,x₁,x₂]): +$growth"),
                listOf("Input channels to layer 3" to (8+growth*2).toDouble(), "Output channels" to (8+growth*3).toDouble()),
                "xₗ = Hₗ([x₀, x₁, …, xₗ₋₁])", "Every layer receives all earlier feature maps by CONCATENATION, not addition.")
        }
        DeepVisualization.Mobilenet -> {
            val channels = control*8; val filters = control*16
            val standard = 3*3*channels*filters; val depthwise = 3*3*channels+channels*filters
            CnnArchitectureFrame(listOf("Input: $channels channels", "Depthwise 3×3 per channel", "Pointwise 1×1 mixing", "Output: $filters channels"),
                listOf("Standard weights" to standard.toDouble(), "Separable weights" to depthwise.toDouble(),
                    "Saving" to (1.0-depthwise.toDouble()/standard)*100),
                "standard = 3² C M; separable = 3² C + C M", "Depthwise spatial filtering and pointwise channel mixing replace one standard convolution.")
        }
        DeepVisualization.Efficientnet -> {
            val phi = control.coerceIn(2,9)/2.0
            val depth = 1.2.pow(phi); val width = 1.1.pow(phi); val resolution = 1.15.pow(phi)
            CnnArchitectureFrame(listOf("MBConv block", "Depth × %.2f".format(depth), "Width × %.2f".format(width),
                "Resolution × %.2f".format(resolution), "Scaled network"),
                listOf("Depth factor" to depth, "Width factor" to width, "Resolution factor" to resolution),
                "compute ≈ depth × width² × resolution²", "Compound scaling changes all three axes together rather than only making a network deeper.")
        }
        DeepVisualization.Convnext -> CnnArchitectureFrame(
            listOf("Input", "7×7 depthwise Conv", "LayerNorm", "1×1 expand ×4", "GELU", "1×1 project", "Residual add"),
            listOf("Depthwise kernel" to 7.0, "Expansion ratio" to 4.0),
            "block(x) = x + PW₂(GELU(PW₁(LN(DW₇×₇(x)))))", "Modern convolutional block uses large depthwise kernels and pointwise expansion.")
        else -> error("Not a CNN architecture")
    }
}

@Composable
internal fun StageThreeCnnArchitectureScreen(topic: LearnTopic, kind: DeepVisualization) {
    var selected by remember(topic.id) { mutableIntStateOf(0) }
    var control by remember(topic.id) { mutableIntStateOf(4) }
    val frame = remember(kind, control) { StageThreeCnnArchitectureEngine.frame(kind, control) }
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        item { SectionTitle(kind.title, "Architecture and the computation that makes it distinctive") }
        item { GlassPanel(Modifier.fillMaxWidth()) {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("Information flow", color = LabText, fontWeight = FontWeight.Bold)
                ArchitectureDrawing(kind, selected)
                Text(frame.distinction, color = LabCyan, fontSize = 12.sp)
            }
        } }
        item { GlassPanel(Modifier.fillMaxWidth()) {
            Column(verticalArrangement = Arrangement.spacedBy(7.dp)) {
                Text("Inspect a stage", color = LabText, fontWeight = FontWeight.Bold)
                frame.stages.forEachIndexed { i, stage -> SegmentedOption("${i+1}. $stage", i == selected, Modifier.fillMaxWidth()) { selected = i } }
            }
        } }
        item { GlassPanel(Modifier.fillMaxWidth()) {
            Column(verticalArrangement = Arrangement.spacedBy(7.dp)) {
                Text("Computed example", color = LabText, fontWeight = FontWeight.Bold)
                frame.numbers.forEach { (label, value) -> Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                    Text(label, color = LabMuted, fontSize = 12.sp)
                    Text("%.2f".format(value), color = LabCyan, fontSize = 12.sp)
                } }
                Text(frame.equation, color = LabOrange, fontSize = 12.sp)
                if (kind in setOf(DeepVisualization.Vgg, DeepVisualization.Resnet, DeepVisualization.Densenet,
                        DeepVisualization.Mobilenet, DeepVisualization.Efficientnet)) {
                    Text(when (kind) {
                        DeepVisualization.Vgg -> if (control % 2 == 0) "VGG16" else "VGG19"
                        DeepVisualization.Resnet -> "Input x = %.1f".format(control/10.0)
                        DeepVisualization.Densenet -> "Growth rate = $control"
                        DeepVisualization.Mobilenet -> "Base channels = ${control*8}"
                        else -> "Compound scale φ = %.1f".format(control/2.0)
                    }, color = LabText, fontSize = 12.sp)
                    Slider(control.toFloat(), { control = it.toInt().coerceIn(2,9) }, valueRange = 2f..9f)
                }
            }
        } }
    }
}

@Composable private fun ArchitectureDrawing(kind: DeepVisualization, selected: Int) {
    Canvas(Modifier.fillMaxWidth().height(190.dp).background(Color(0xFF071B37))) {
        val w = size.width; val h = size.height
        val cyan = LabCyan; val purple = LabPurple
        fun block(x: Float, y: Float, width: Float, height: Float, highlight: Boolean) {
            drawRect(if (highlight) cyan.copy(alpha=.55f) else purple.copy(alpha=.32f), Offset(x,y), Size(width,height))
            drawRect(if (highlight) cyan else purple, Offset(x,y), Size(width,height), style=Stroke(width=2f))
        }
        when (kind) {
            DeepVisualization.Inception -> {
                block(w*.03f,h*.37f,w*.15f,h*.26f,selected==0)
                for (i in 0..3) {
                    val y=h*(.05f+i*.235f)
                    drawLine(cyan,Offset(w*.18f,h*.5f),Offset(w*.25f,y+h*.08f),2f)
                    block(w*.25f,y,w*.45f,h*.16f,selected==i+1)
                    drawLine(cyan,Offset(w*.7f,y+h*.08f),Offset(w*.78f,h*.5f),2f)
                }
                block(w*.78f,h*.37f,w*.18f,h*.26f,selected==5)
            }
            DeepVisualization.Resnet -> {
                block(w*.05f,h*.43f,w*.17f,h*.2f,selected==0)
                block(w*.38f,h*.43f,w*.23f,h*.2f,selected==1)
                block(w*.78f,h*.43f,w*.17f,h*.2f,selected==4)
                drawLine(cyan,Offset(w*.22f,h*.53f),Offset(w*.38f,h*.53f),3f)
                drawLine(cyan,Offset(w*.61f,h*.53f),Offset(w*.78f,h*.53f),3f)
                drawLine(LabOrange,Offset(w*.22f,h*.53f),Offset(w*.27f,h*.16f),3f)
                drawLine(LabOrange,Offset(w*.27f,h*.16f),Offset(w*.72f,h*.16f),3f)
                drawLine(LabOrange,Offset(w*.72f,h*.16f),Offset(w*.72f,h*.53f),3f)
                drawCircle(LabOrange,8f,Offset(w*.72f,h*.53f))
            }
            DeepVisualization.Densenet -> {
                for (i in 0..3) {
                    val x=w*(.06f+i*.23f)
                    block(x,h*.5f,w*.17f,h*.23f,selected==i)
                    for (j in i+1..3) drawLine(if (j-i==1) cyan else LabOrange, Offset(x+w*.17f,h*.61f),
                        Offset(w*(.06f+j*.23f),h*.61f-(j-i-1)*14),2f)
                }
            }
            DeepVisualization.Mobilenet -> {
                block(w*.04f,h*.36f,w*.24f,h*.31f,selected==0)
                block(w*.38f,h*.36f,w*.24f,h*.31f,selected==1)
                block(w*.72f,h*.36f,w*.24f,h*.31f,selected==2)
                drawLine(cyan,Offset(w*.28f,h*.52f),Offset(w*.38f,h*.52f),3f)
                drawLine(LabOrange,Offset(w*.62f,h*.52f),Offset(w*.72f,h*.52f),3f)
            }
            else -> {
                val count = if (kind==DeepVisualization.Lenet) 6 else 5
                for (i in 0 until count) {
                    val x=w*(.04f+i*.92f/count)
                    block(x,h*(.2f+i*.06f),w*.72f/count,h*(.62f-i*.1f),selected==i)
                    if (i<count-1) drawLine(cyan,Offset(x+w*.72f/count,h*.51f),Offset(x+w*.92f/count,h*.51f),2f)
                }
            }
        }
    }
}
