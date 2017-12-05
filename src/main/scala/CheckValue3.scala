import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by lanjianchao on 2017/11/5.
  */
object CheckValue3 {
  def main(args: Array[String]) {
    //创建入口对象
    val conf = new SparkConf().setAppName("CheckValue3").setMaster("local")
    val sc = new SparkContext(conf)
    val rawData = sc.textFile("/Users/lanjianchao/Movies/kddcup.data_10_percent.csv")

    val LabelsAndData = rawData.map {
      //代码块执行RDD[String] => RDD[Vector]
      line =>
        //toBuffer创建一个可变列表(Buffer)
        val buffer = line.split(",").toBuffer
        val label = buffer.remove(buffer.length - 1)
        val vector = Vectors.dense(buffer.map(_.toDouble).toArray)
        (label, vector)
    }
    val data = LabelsAndData.values.cache() //转化值并进行缓存
    println(LabelsAndData.values)
    //建立kmeansModel
    val kmeans = new KMeans()
    val model = kmeans.run(data)
    model.clusterCenters.foreach(println)

    /** 由CheckValue1已知该数据集有23类，CheckValue2分类肯定不准确，所以下面我利用给定的类别标号信息来
      * 直观的看到分好的簇中包含哪些类型的样本，对每个簇中的标号进行计数，并以可读的方式输出
      */
    //对标号进行计数
    val clusterLabelCount = LabelsAndData.map {
      case (label, datum) =>
        val cluster = model.predict(datum)
        (cluster, label)
    }.countByValue()
    //将簇-类别进行计数，输出
    println("计数结果如下")
    clusterLabelCount.toSeq.sorted.foreach {
      case ((cluster, label), count) =>
        //使用字符插值器对变量的输出进行格式化
        println(f"$cluster%1s$label%18s$count%8s")
    }
  }
}
