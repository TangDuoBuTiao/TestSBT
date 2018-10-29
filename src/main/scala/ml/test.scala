package ml

import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.{SparkConf, SparkContext}

object test extends App {

  val conf = new SparkConf().setAppName("DecisionTree").setMaster("spark://sict136:7077")
  val sc = new SparkContext(conf)
  sc.setLogLevel("ERROR")

  val data = sc.textFile("/user/Tian/data/kddcup.data").map{
    x =>
      val tokens = x.split(",",-1)
      val label = tokens.last.toDouble
      val features = tokens.dropRight(1)
      new LabelPoint(label,Vectors.dense(features))
  }

}
