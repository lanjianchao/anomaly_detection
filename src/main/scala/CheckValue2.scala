import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by lanjianchao on 2017/11/5.
  */
object CheckValue2 {
  def main(args: Array[String]) {

    //创建入口对象
    val conf = new SparkConf().setAppName("CheckValue2").setMaster("spark://se018:7077")
    val sc = new SparkContext(conf)
    val rawData = sc.textFile("/home/cs/jclan/kddcup.data_10_percent.csv")
    var clusterIndex: Int = 0

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

    //建立kmeansModel
    val kmeans = new KMeans()
    kmeans.setK(30)
    kmeans.setMaxIterations(10)
    kmeans.setRuns(3)
    val model = kmeans.run(data)
    model.save(sc,"/Users/lanjianchao/Movies/model.txt")

    //    model.clusterCenters.foreach(x => {
    //      println("Center Point of Cluster " + clusterIndex + ":")
    //      println(x)
    //      clusterIndex += 1})

    val rawData1 = sc.textFile("/Users/lanjianchao/Movies/test.txt")
    val parsedTestData = rawData1.map(line => {
      Vectors.dense(line.split(",").map(_.trim).filter(!"".equals(_)).map(_.toDouble))

    })
    parsedTestData.collect().foreach(testDataLine => {
      val predictedClusterIndex:
        Int = model.predict(testDataLine)
      println("The data " + testDataLine.toString + " belongs to cluster " +
        predictedClusterIndex)
    })
  }

}
