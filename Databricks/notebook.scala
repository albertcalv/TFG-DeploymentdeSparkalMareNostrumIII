// Databricks notebook source exported at Tue, 12 Apr 2016 20:43:21 UTC
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

import org.apache.spark.mllib.clustering.{KMeansModel, KMeans}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD

import scala.collection.mutable

import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.random._
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.configuration.{Algo, FeatureType}
import org.apache.spark.mllib.tree.model.{Split, DecisionTreeModel, Node, Predict}
import org.apache.spark.rdd.{PairRDDFunctions, RDD}
import org.apache.spark.SparkContext

import scala.util.Random
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}

// COMMAND ----------

def generateKMeansVectors(
      sc: SparkContext,
      numRows: Long,
      numCols: Int,
      numCenters: Int,
      numPartitions: Int,
      seed: Long = System.currentTimeMillis()): RDD[Vector] = {
          RandomRDDs.randomRDD(sc, new KMeansDataGenerator(numCenters, numCols, seed), numRows, numPartitions, seed)
      }

class KMeansDataGenerator(
  val numCenters: Int,
  val numColumns: Int,
  val seed: Long) extends RandomDataGenerator[Vector] {

  private val rng = new java.util.Random(seed)
  private val rng2 = new java.util.Random(seed + 24)
  private val scale_factors = Array.fill(numCenters)(rng.nextInt(20) - 10)
  private val concentrations: Seq[Double] = {
    val rand = Array.fill(numCenters)(rng.nextDouble())
    val randSum = rand.sum
    val scaled = rand.map(x => x / randSum)
    (1 to numCenters).map{i =>
      scaled.slice(0, i).sum
    }
  }
  private val centers = (0 until numCenters).map{i =>
    Array.fill(numColumns)((2 * rng.nextDouble() - 1)*scale_factors(i))
  }
  override def nextValue(): Vector = {
    val pick_center_rand = rng2.nextDouble()
    val centerToAddTo = centers(concentrations.indexWhere(p => pick_center_rand <= p))
    Vectors.dense(Array.tabulate(numColumns)(i => centerToAddTo(i) + rng2.nextGaussian()))
  }
  override def setSeed(seed: Long) {
    rng.setSeed(seed)
  }
  override def copy(): KMeansDataGenerator = new KMeansDataGenerator(numCenters, numColumns, seed)
}


// COMMAND ----------

val points = 10
val dimensions = 10
val numCenters = 5
val numPartitions = 10
val seed = 5
val iterations = 10 

val rdd = generateKMeansVectors(sc, points, dimensions, numCenters,numPartitions, seed)
rdd.foreach(println)
val kmeans = KMeans.train(rdd, numCenters, iterations)
val WSSSE = kmeans.computeCost(rdd)
println("Error:" + WSSSE )

//display(kmeans, rdd, "Test kmeans")


// COMMAND ----------

val points = 10000000
val dimensions = 10
val numCenters = 5
val numPartitions = 10
val seed = 5
val iterations = 10 

val rdd = generateKMeansVectors(sc, points, dimensions, numCenters,numPartitions, seed)

val kmeans = KMeans.train(rdd, numCenters, iterations)
val WSSSE = kmeans.computeCost(rdd)
println("Error:" + WSSSE )



// COMMAND ----------

val points = 10000000
val dimensions = 10
val numCenters = 5
val numPartitions = 10
val seed = 5
val iterations = 10 

val rdd = generateKMeansVectors(sc, points, dimensions, numCenters,numPartitions, seed)

val kmeans = KMeans.train(rdd, numCenters, iterations)
val WSSSE = kmeans.computeCost(rdd)
println("Error:" + WSSSE )



// COMMAND ----------

val points = 100000000
val dimensions = 1
val numCenters = 5
val numPartitions = 10
val seed = 5
val iterations = 10 

val rdd = generateKMeansVectors(sc, points, dimensions, numCenters,numPartitions, seed)

val kmeans = KMeans.train(rdd, numCenters, iterations)
val WSSSE = kmeans.computeCost(rdd)
println("Error:" + WSSSE )



