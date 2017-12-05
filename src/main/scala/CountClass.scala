import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD

/**
  * Created by lanjianchao on 2017/11/5.
  */
object CountClass {
  /**
    * 欧氏距离公式
    * x.toArray.zip(y.toArray)对应 "两个向量相应元素"
    * map(p => p._1 - p._2)对应 "差"
    * map(d => d*d).sum对应 "平方和"
    * math.sqrt()对应 "平方根"
    *
    * @param x
    * @param y
    * @return
    */
  def distance(x: Vector, y: Vector) = {
    math.sqrt(x.toArray.zip(y.toArray).map(p => p._1 - p._2).map(d => d*d).sum)
  }

  /**
    * 欧氏距离公式应用到model中
    * KMeansModel.predict方法中调用了KMeans对象的findCloest方法
    *
    * @param datum
    * @param model
    * @return
    */
  def distToCentroid(datum: Vector, model: KMeansModel) = {
    //找最短距离的点
    val cluster = model.predict(datum)
    //找中心点
    val centroid = model.clusterCenters(cluster)
    distance(centroid, datum)
  }

  /**
    * k值model平均质心距离
    *
    * @param data RDD向量格式
    * @param k  分类数
    * @return
    */
  def clusteringScore(data: RDD[Vector], k: Int) = {
    val kmeans = new KMeans()
    kmeans.setK(k)
    val model = kmeans.run(data)
    data.map(datum => distToCentroid(datum, model)).mean()

  }

  /**
    * 对k的取值进行评价
    * scala通常采用(x to y by z)这种形式建立一个数字集合，该集合的元素为闭合区间的等差数列
    * 这种语法可用于建立一系列k值，然后对每个值分别执行莫项任务
    * @param data
    */
  def check(data: RDD[Vector]) = {
    (5 to 40 by 5).map(k => (k, CountClass.clusteringScore(data, k))).foreach(println)
  }
}
