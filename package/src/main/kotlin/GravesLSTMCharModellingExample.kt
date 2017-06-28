import org.apache.commons.io.FileUtils
import org.deeplearning4j.nn.api.Layer
import org.deeplearning4j.nn.api.Model
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.BackpropType
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.layers.GravesLSTM
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.api.IterationListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.parallelism.ParallelWrapper
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.api.buffer.util.DataTypeUtil
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

import java.io.File
import java.io.IOException
import java.net.URL
import java.nio.charset.Charset
import java.util.Random
import java.util.Arrays

import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import com.google.gson.GsonBuilder

import kotlin.coroutines.experimental.*

object Dammy {
  fun next() = buildSequence {
    val mm    = mutableMapOf<Int, List<Pair<List<Int>,List<Int>>>>()
    val txts  = File("data/DATASET.txt").readText().split("\n").filter { it.length != 0 }
    val size  = txts.size
    var scope = 0
    (0..20000).map {   
      // setting dimention size
      // first  1. batch size
      // second 2. char(or word) dimentions
      // second 3. length of sentence
      // initialize here
      val WIDTH            = 500
      val input : INDArray = Nd4j.zeros( WIDTH, 100, 31 )
      val label : INDArray = Nd4j.zeros( WIDTH, 100, 31 )
      (scope..scope+100).map { count ->
        val hl   = txts[count].split(",")
        val head = hl[0].split(" ").map { it.toInt() }
        val tail = hl[1].split(" ").map { it.toInt() }
        head.mapIndexed { i,j -> 
          input.putScalar( listOf(count-1-scope, j, i).toIntArray(), 1.0 )
        }
        tail.mapIndexed { i,j -> 
          label.putScalar( listOf(count-1-scope, j, i).toIntArray(), 1.0 )
        }
        val last = hl[1].map { it.toInt() }
      }
      scope += WIDTH
      if(scope + WIDTH > size) scope = 0
      val ds = DataSet( input, label )
      println( "shape ${input.shape().toList()}" )
      yield( ds )
    }
  }
}

object GravesLSTMCharModelling {
  fun train() {
    val lstmLayerSize = 200
    val miniBatchSize = 32
    val exampleLength = 1000
    val tbpttLength   = 50
    val numEpochs     = 1
    val generateSamplesEveryNMinibatches = 10
    val nSamplesToGenerate = 4
    val nCharactersToSample = 300
    var generationInitialization:String? = null
    val rng = Random(12345)
    val conf = NeuralNetConfiguration.Builder()
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
      .learningRate(0.1)
      .rmsDecay(0.95)
      .seed(12345)
      .regularization(true)
      .l2(0.001)
         .weightInit(WeightInit.XAVIER)
         .updater(Updater.RMSPROP)
      .list()
      .layer(0, GravesLSTM.Builder()
                  .nIn( 100 )
                  .nOut( lstmLayerSize ) 
                  .activation( Activation.TANH).build() )
      .layer(1, GravesLSTM.Builder()
                  .nIn( lstmLayerSize )
                  .nOut( lstmLayerSize )
                  .activation( Activation.TANH).
                  build() )
      .layer(2, RnnOutputLayer
                  .Builder( LossFunction.MCXENT )
                  .nIn( lstmLayerSize )
                  .nOut( 100 )
                  .activation( Activation.SOFTMAX )
                  .build() )
      .backpropType(BackpropType.TruncatedBPTT)
          .tBPTTForwardLength(tbpttLength)
          .tBPTTBackwardLength(tbpttLength)
      .pretrain(false)
      .backprop(true)
      .build()
    val net = MultiLayerNetwork(conf)
    net.init()
    net.setListeners( ScoreIterationListener(1) )
    for( ds in Dammy.next() ) { 
      net.fit( ds )
    }
    val layers = net.getLayers()
    var totalNumParams = 0
    for( i in (0..layers.size-1)) {
      val nParams = layers[i].numParams()
      println("Number of parameters in layer $i $nParams")
      totalNumParams += nParams
    }
    println("Total number of network parameters: $totalNumParams")
    println("Complete");
  }
	@JvmStatic
  fun main( args : Array<String> ) {
    val argments = args.toList()
    when { 
      argments.contains("train") -> train()
      else                       -> train()
    }
	}
}
