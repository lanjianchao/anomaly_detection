import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by lanjianchao on 2017/11/5.
  */
object CheckValue4 {
  def main(args: Array[String]) {

    //创建入口对象
    val conf = new SparkConf().setAppName("CheckValue1").setMaster("local")
    val sc= new SparkContext(conf)
    val rawData = sc.textFile("/Users/lanjianchao/Movies/kddcup.data_10_percent.csv")

    val LabelsAndData = rawData.map{   //代码块执行RDD[String] => RDD[Vector]
      line =>
        //toBuffer创建一个可变列表(Buffer)
        val buffer = line.split(",").toBuffer
        buffer.remove(1, 3)
        val label = buffer.remove(buffer.length-1)
        val vector = Vectors.dense(buffer.map(_.toDouble).toArray)
        (label, vector)
    }
    val data = LabelsAndData.values.cache()  //转化值并进行缓存


    CountClass.check(data)     //给k的取值进行评价,k=(5,10,15,20,25,30,35,40)
    //    运行结果:
    //    (5,1938.8583418059188)
    //    (10,1629.469780026074)
    //    (15,1380.2560462290849)
    //    (20,1309.6468484397622)
    //    (25,1041.0183009597501)
    //    (30,1007.0769941770079)
    //    (35,562.989358358847)
    //    (40,421.86047502003527)
  }
}
