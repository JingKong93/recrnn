import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.nn.{Sequential => _}
import com.intel.analytics.bigdl.nn._
val arr = Array(Tensor(T(
  T(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3629.0)
  ,T(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 4.0, 3629.0))))

val label2 = Array(Tensor(1).fill(1),Tensor(1).fill(1))


val sam = Sample(arr,label2)


val features = Array(Tensor(2, 2).rand, Tensor(2, 2).rand)
val labels = Array(Tensor(1).fill(1), Tensor(1).fill(-1))
val sample = Sample(features, labels)
sample.label(1)
import com.intel.analytics.bigdl.numeric.NumericFloat
val seed = 100
//RNG.setSeed(seed)
val inputFrameSize = 5
val outputFrameSize = 3
val kW = 5
val dW = 2
val layer = TemporalConvolution(20, outputFrameSize,4)

//Random.setSeed(seed)
val input = Tensor(6, 20)


val output = layer.updateOutput(input)
import java.io.File


new File(s"jjj.zip").delete()