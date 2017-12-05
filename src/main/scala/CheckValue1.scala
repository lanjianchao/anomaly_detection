import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by lanjianchao on 2017/11/5.
  */
object CheckValue1 {
  def main(args: Array[String]) {

    //创建入口对象
    val conf = new SparkConf().setAppName("CheckValue1").setMaster("local")
    val sc = new SparkContext(conf)
    val rawData = sc.textFile("/Users/lanjianchao/Movies/kddcup.data_10_percent.csv")

    /**
      * 实验一
      * 分类统计样本个数，降序排序
      */
    val sort_result = rawData.map(_.split(",").last).countByValue().toSeq.sortBy(_._2).reverse
    sort_result.foreach(println)
  }
}
