import org.apache.commons.io.FileUtils
import org.deeplearning4j.nn.api.Layer
import org.deeplearning4j.nn.api.Model
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.BackpropType
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.layers.GravesLSTM
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.api.IterationListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.parallelism.ParallelWrapper
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
    val mm   = mutableMapOf<Int, List<Pair<List<Int>,List<Int>>>>()
    val txts = File("data/DATASET.txt").readText().split("\n").filter { it.length != 0 }
    val size = txts.size
    var scope = 0
    while(true) {   
      // setting dimention size
      // first  1. batch size
      // second 2. char(or word) dimentions
      // second 3. length of sentence
      // initialize here
      val input : INDArray = Nd4j.zeros( 32, 100, 31)
      val label : INDArray = Nd4j.zeros( 32, 100, 31)
      (scope..scope+31).map { count ->
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
      scope += 32
      if(scope + 32> size)
        scope = 0
      val ds = DataSet( input, label )
      println( "shape ${input.shape().toList()}" )
      yield( ds )
    }
  }
}
object GravesLSTMCharModelling {
	@JvmStatic
  fun main( args : Array<String> ) {
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
			.layer(0, GravesLSTM.Builder().nIn( 100 ).nOut(lstmLayerSize)
				.activation(Activation.TANH).build())
			.layer(1, GravesLSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
				.activation(Activation.TANH).build())
			.layer(2, RnnOutputLayer.Builder(LossFunction.MCXENT).activation(Activation.SOFTMAX)
				.nIn(lstmLayerSize).nOut( 100 ).build())
        .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(tbpttLength).tBPTTBackwardLength(tbpttLength)
			.pretrain(false).backprop(true)
			.build()

		val net = MultiLayerNetwork(conf)
		net.init()
		net.setListeners( ScoreIterationListener(1) )

    for( ds in Dammy.next() ) { 
      net.fit( ds )
    }
    /*
		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();
		net.setListeners(new ScoreIterationListener(1), new IterationListener() {
            @Override
            public boolean invoked() {
                return true;
            }

            @Override
            public void invoke() {

            }

            @Override
            public void iterationDone(Model model, int iteration) {
                System.out.println("--------------------");
                System.out.println("Sampling characters from network given initialization \"" + (generationInitialization == null ? "" : generationInitialization) + "\"");
                String[] samples = sampleCharactersFromNetwork(generationInitialization, (MultiLayerNetwork) model,iter,rng,nCharactersToSample,nSamplesToGenerate);
                for( int j = 0; j<samples.length; j++) {
                    System.out.println("----- Sample " + j + " -----");
                    System.out.println(samples[j]);
                    System.out.println();
                }
            }
        });


        //Print the  number of parameters in the network (and for each layer)
		Layer[] layers = net.getLayers();
		int totalNumParams = 0;
		for( int  i= 0; i < layers.length; i++) {
			int nParams = layers[i].numParams();
			System.out.println("Number of parameters in layer " + i + ": " + nParams);
			totalNumParams += nParams;
		}

		System.out.println("Total number of network parameters: " + totalNumParams);


        // ParallelWrapper will take care of load balancing between GPUs.
        ParallelWrapper wrapper = new ParallelWrapper.Builder(net)
            // DataSets prefetching options. Set this value with respect to number of actual devices
            .prefetchBuffer(24)

            // set number of workers equal or higher then number of available devices. x1-x2 are good values to start with
            .workers(4)

            // rare averaging improves performance, but might reduce model accuracy
            .averagingFrequency(3)

            // if set to TRUE, on every averaging model score will be reported
            .reportScoreAfterAveraging(true)

            // optinal parameter, set to false ONLY if your system has support P2P memory access across PCIe (hint: AWS do not support P2P)
            .useLegacyAveraging(true)

            .build();

        wrapper.fit(iter);


		System.out.println("\n\nExample complete");
    */
	}
}
