import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by lanjianchao on 2017/11/5.
  */
object CheckValue5 {
  def main(args: Array[String]) {

    //创建入口对象
    val conf = new SparkConf().setAppName("CheckValue5").setMaster("spark://se018:7077")
    val sc= new SparkContext(conf)
    val rawData = sc.textFile("/home/cs/jclan/kddcup.data_10_percent.csv")

    val LabelsAndData = rawData.map{   //代码块执行RDD[String] => RDD[Vector]
      line =>
        //toBuffer创建一个可变列表(Buffer)
        val buffer = line.split(",").toBuffer
        val label = buffer.remove(buffer.length-1)
        val vector = Vectors.dense(buffer.map(_.toDouble).toArray)
        (label, vector)
    }
    val data = LabelsAndData.values.cache()  //转化值并进行缓存

    //建立kmeansModel
    val kmeans = new KMeans()
    //设置给定k值的运行次数
    kmeans.setRuns(5)
    kmeans.setMaxIterations(40)
    kmeans.setK(20)
    val model = kmeans.run(data)
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
//    kmeans.setEpsilon(1.0e-6)
//    (10 to 40 by 10).par.map(k => (k, CountClass.clusteringScore(data, k))).toList.foreach(println)

    //    程序运行如下:
    //          (30,584.3824466136748)
    //          (40,473.13630965059355)
    //          (50,397.1680468789708)
    //          (60,224.7032729131013)
    //          (70,209.75091102083454)
    //          (80,189.24155085526263)
    //          (90,192.57698780271707)
    //          (100,119.81903683729702)

    /**
      * 总结：随着k的增大，结果得分持续下降，我们要找到k值的临界点，过了这个临界点之后继续增加k值并不会显著降低得分
      * 这个点的k值-得分曲线的拐点。这条曲线通常在拐点之后会继续下行但最终趋于水平。
      * 在本实例中k>100之后得分下降很明显，故得出k的拐点应该大于100
      */


  }
}
