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
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.GenShape
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.PhaseNineEngines
import kotlin.math.abs
import kotlin.math.exp

internal data class GenerativeVariantFrame(val flow:List<String>,val values:List<Pair<String,Double>>,
                                           val explanation:String,val equation:String)

internal object StageThreeGenerativeVariantEngine {
    fun sparseActivations(control:Int):List<Double> {
        val hidden=PhaseNineEngines.autoencoder(GenShape.Circle,latentDims=2).hidden
        val threshold=.5+control*.05
        return hidden.mapIndexed { i,value -> if(i%4==0) value else (value-threshold).coerceAtLeast(0.0) }
    }
    fun frame(kind:DeepVisualization,control:Int):GenerativeVariantFrame {
        val x=control/10.0
        val auto=PhaseNineEngines.autoencoder(GenShape.Circle,latentDims=2)
        return when(kind) {
            DeepVisualization.SparseAe -> {
                val sparse=sparseActivations(control)
                val active=sparse.count { it>1e-6 }
                GenerativeVariantFrame(listOf("Input image","Encoder","Sparse latent activations","Decoder","Reconstruction"),
                    listOf("Active latent units" to active.toDouble(),"Sparsity penalty" to sparse.sum()*x,
                        "Reconstruction loss" to auto.loss),
                    "A sparsity penalty pushes most latent activations toward zero.","loss = reconstruction + λ Σ|latent activation|")
            }
            DeepVisualization.ConvAe -> GenerativeVariantFrame(
                listOf("8×8 image","3×3 Conv feature maps","4×4 downsample","Latent tensor","Upsample","3×3 Conv","Reconstruction"),
                listOf("Input pixels" to 64.0,"Latent cells" to 16.0,"Reconstruction MSE" to auto.loss),
                "Spatial feature maps are reduced then upsampled; the reconstruction retains image layout.",
                "Conv → downsample → latent map → upsample → Conv")
            DeepVisualization.Dcgan -> GenerativeVariantFrame(
                listOf("Noise z","Dense projection","Transpose Conv upsample","Fake image","Conv discriminator","Real/fake score"),
                listOf("Generator scale" to x,"Discriminator score" to sigmoid(1.2-2*x)),
                "Generator expands a latent tensor into image pixels; discriminator reduces images with convolutions.",
                "min G max D: log D(real) + log(1−D(G(z)))")
            DeepVisualization.ConditionalGan -> GenerativeVariantFrame(
                listOf("Noise z + class 7","Conditional generator","Digit 7 sample","Image + class 7","Conditional discriminator"),
                listOf("Class condition" to 7.0,"D(fake | class 7)" to sigmoid(1.5-2*x)),
                "The class label enters BOTH generator and discriminator, so realism is judged relative to the requested class.",
                "G(z, y); D(image, y)")
            DeepVisualization.CycleGan -> {
                val forward=x*.85+.1; val backward=forward*.8+.1
                GenerativeVariantFrame(listOf("Domain A","G: A→B","Domain B","F: B→A","Cycle reconstruction A"),
                    listOf("A value" to x,"G(A)" to forward,"F(G(A))" to backward,"Cycle L1" to abs(x-backward)),
                    "Two generators translate in opposite directions. Cycle loss checks whether returning recovers the source.",
                    "Lcycle = |F(G(A))−A| + |G(F(B))−B|")
            }
            DeepVisualization.StyleGan -> GenerativeVariantFrame(
                listOf("z noise","Mapping network → w","Coarse style","Middle style","Fine style","Image"),
                listOf("Coarse style" to x,"Middle style" to (.5+x*.4),"Fine style" to (1-x*.3)),
                "A learned style vector modulates multiple synthesis layers at different spatial scales.",
                "w = mapping(z); featureₗ = modulatedConv(featureₗ₋₁, styleₗ)")
            DeepVisualization.Wgan -> {
                val real=1.2+x*.2; val fake=.4+x*.4
                GenerativeVariantFrame(listOf("Real samples","Critic scores","Fake samples","Score gap","Generator update"),
                    listOf("Critic(real)" to real,"Critic(fake)" to fake,"Score gap" to real-fake),
                    "The critic outputs unrestricted scores, not real/fake probabilities. The gap estimates Wasserstein distance.",
                    "critic objective = E[D(real)] − E[D(fake)]")
            }
            DeepVisualization.LatentDiffusion -> {
                val diffusion=PhaseNineEngines.diffusion(GenShape.Circle,step=control*2,totalSteps=24)
                GenerativeVariantFrame(listOf("Image","VAE encoder","Compact latent z","Noise latent","Denoise latent","VAE decoder","Image"),
                    listOf("Pixel cells" to 64.0,"Latent dimensions" to 2.0,"Noise step" to (control*2).toDouble(),
                        "Denoising loss" to diffusion.loss),
                    "The noising and denoising happen in a compressed latent representation, then a decoder returns to pixel space.",
                    "image → E(image)=z → diffusion(z) → D(z₀)")
            }
            else -> error("Not a generative variant")
        }
    }
    private fun sigmoid(x:Double)=1.0/(1.0+exp(-x))
}

@Composable
internal fun StageThreeGenerativeVariantsScreen(topic:LearnTopic,kind:DeepVisualization) {
    var control by remember(topic.id){mutableIntStateOf(4)}
    var stage by remember(topic.id){mutableIntStateOf(0)}
    val frame=remember(kind,control){StageThreeGenerativeVariantEngine.frame(kind,control)}
    LazyColumn(Modifier.fillMaxSize().padding(16.dp),verticalArrangement=Arrangement.spacedBy(12.dp)) {
        item { SectionTitle(kind.title,"Inspect the data flow and computed teaching example") }
        item { GlassPanel(Modifier.fillMaxWidth()) {
            Column(verticalArrangement=Arrangement.spacedBy(8.dp)) {
                Text("Computational path",color=LabText,fontWeight=FontWeight.Bold)
                VariantFlow(kind,frame.flow,stage,control)
                frame.flow.forEachIndexed { i,label -> SegmentedOption("${i+1}. $label",i==stage,Modifier.fillMaxWidth()){stage=i} }
            }
        } }
        item { GlassPanel(Modifier.fillMaxWidth()) {
            Column(verticalArrangement=Arrangement.spacedBy(7.dp)) {
                Text("What changes",color=LabText,fontWeight=FontWeight.Bold)
                Text(frame.explanation,color=LabMuted,fontSize=12.sp)
                frame.values.forEach { (label,value) -> Row(Modifier.fillMaxWidth(),horizontalArrangement=Arrangement.SpaceBetween) {
                    Text(label,color=LabMuted,fontSize=12.sp); Text("%.3f".format(value),color=LabCyan,fontSize=12.sp)
                } }
                Text(frame.equation,color=LabOrange,fontSize=12.sp)
                Text("Example control %.1f".format(control/10.0),color=LabText,fontSize=12.sp)
                Slider(control.toFloat(),{control=it.toInt().coerceIn(2,9)},valueRange=2f..9f)
                Text("Synthetic educational calculation; no large model is trained on device.",color=LabMuted,fontSize=11.sp)
            }
        } }
        if(kind==DeepVisualization.SparseAe||kind==DeepVisualization.ConvAe||kind==DeepVisualization.LatentDiffusion) item {
            val image=PhaseNineEngines.shapeImage(GenShape.Circle)
            GlassPanel(Modifier.fillMaxWidth()) {
                Column(verticalArrangement=Arrangement.spacedBy(7.dp)) {
                    Text("Input image pixels",color=LabText,fontWeight=FontWeight.Bold)
                    TinyImageGrid(image)
                }
            }
        }
    }
}

@Composable private fun VariantFlow(kind:DeepVisualization,stages:List<String>,selected:Int,control:Int) {
    Canvas(Modifier.fillMaxWidth().height(155.dp).background(Color(0xFF071B37))) {
        val w=size.width; val h=size.height
        when(kind) {
            DeepVisualization.CycleGan -> {
                val a=Offset(w*.13f,h*.5f); val b=Offset(w*.87f,h*.5f)
                drawLine(LabCyan,a,Offset(w*.5f,h*.18f),4f)
                drawLine(LabCyan,Offset(w*.5f,h*.18f),b,4f)
                drawLine(LabOrange,b,Offset(w*.5f,h*.82f),4f)
                drawLine(LabOrange,Offset(w*.5f,h*.82f),a,4f)
                drawCircle(LabCyan,15f,a);drawCircle(LabOrange,15f,b)
                drawCircle(LabPurple,10f,Offset(w*.5f,h*.18f));drawCircle(LabPurple,10f,Offset(w*.5f,h*.82f))
            }
            DeepVisualization.StyleGan -> {
                for(i in 0..3) {
                    val x=w*(.1f+i*.22f)
                    drawRect(if(i==selected%4)LabCyan else LabPurple,Offset(x,h*(.2f+i*.1f)),Size(w*.14f,h*(.6f-i*.2f)))
                    if(i>0)drawLine(LabOrange,Offset(x-w*.08f,h*.1f),Offset(x+w*.07f,h*(.2f+i*.1f)),2f)
                }
            }
            DeepVisualization.Wgan -> {
                for(i in 0..5) {
                    val x=w*(.08f+i*.14f)
                    drawLine(LabCyan,Offset(x,h*.9f),Offset(x,h*(.15f+i*.04f)),8f)
                    drawLine(LabOrange,Offset(x+w*.05f,h*.9f),Offset(x+w*.05f,h*(.55f-i*.035f)),8f)
                }
            }
            DeepVisualization.ConditionalGan -> {
                drawCircle(LabCyan,13f,Offset(w*.12f,h*.32f))
                drawCircle(LabOrange,13f,Offset(w*.12f,h*.72f))
                drawLine(LabCyan,Offset(w*.12f,h*.32f),Offset(w*.43f,h*.5f),3f)
                drawLine(LabOrange,Offset(w*.12f,h*.72f),Offset(w*.43f,h*.5f),3f)
                drawRect(LabPurple,Offset(w*.4f,h*.37f),Size(w*.19f,h*.26f))
                drawLine(LabCyan,Offset(w*.59f,h*.5f),Offset(w*.82f,h*.5f),3f)
                drawLine(LabOrange,Offset(w*.12f,h*.72f),Offset(w*.82f,h*.5f),2f)
                drawRect(LabCyan,Offset(w*.8f,h*.38f),Size(w*.14f,h*.24f),style=androidx.compose.ui.graphics.drawscope.Stroke(3f))
            }
            DeepVisualization.LatentDiffusion -> {
                for(i in 0..4) {
                    val x=w*(.06f+i*.19f);val side=if(i==0||i==4)h*.47f else h*.22f
                    drawRect(if(i==2)LabOrange else if(i==0||i==4)LabCyan else LabPurple,
                        Offset(x,h*.5f-side/2),Size(w*.13f,side),style=androidx.compose.ui.graphics.drawscope.Stroke(3f))
                    if(i>0)drawLine(LabOrange,Offset(x-w*.06f,h*.5f),Offset(x,h*.5f),2f)
                }
            }
            DeepVisualization.ConvAe -> {
                for(i in 0..4) {
                    val x=w*(.08f+i*.19f);val side=h*(if(i<=2) .6f-i*.19f else .22f+(i-2)*.19f)
                    drawRect(if(i==2)LabOrange else LabCyan,Offset(x,h*.5f-side/2),Size(w*.12f,side),
                        style=androidx.compose.ui.graphics.drawscope.Stroke(3f))
                    if(i>0)drawLine(LabPurple,Offset(x-w*.07f,h*.5f),Offset(x,h*.5f),2f)
                }
            }
            DeepVisualization.SparseAe -> {
                val sparse=StageThreeGenerativeVariantEngine.sparseActivations(control)
                sparse.forEachIndexed { i,value ->
                    val x=w*(.03f+i*.058f)
                    val height=(value*h*.8).toFloat().coerceAtLeast(2f)
                    drawRect(if(value>.01)LabCyan else LabMuted,Offset(x,h*.85f-height),Size(w*.035f,height))
                }
            }
            else -> {
                // DCGAN expands maps in G, then contracts them in D.
                val sizes=listOf(.13f,.23f,.38f,.6f,.38f,.2f)
                for(i in stages.indices) {
                    val x=(i+.5f)*w/stages.size
                    val side=h*sizes[i.coerceIn(sizes.indices)]
                    drawRect(if(i==selected)LabCyan else LabPurple,Offset(x-w*.04f,h*.5f-side/2),Size(w*.08f,side))
                    if(i>0)drawLine(LabOrange,Offset(x-w/stages.size*.7f,h*.5f),Offset(x-w*.05f,h*.5f),2f)
                }
            }
        }
    }
}

@Composable private fun TinyImageGrid(image:List<List<Double>>) {
    Canvas(Modifier.fillMaxWidth().height(180.dp).background(Color(0xFF071B37))) {
        val cw=size.width/image.first().size; val ch=size.height/image.size
        image.forEachIndexed { r,row -> row.forEachIndexed { c,value ->
            drawRect(LabCyan.copy(alpha=(.12+.8*value).toFloat()),Offset(c*cw+1,r*ch+1),Size(cw-2,ch-2))
        } }
    }
}
