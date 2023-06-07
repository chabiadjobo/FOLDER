import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.functions;

public class monpremier {
  public static void main(String[] args) {
	SparkSession spark = SparkSession.builder()
    		.appName("RowDataset")  // Définit le nom de l'application Spark
    		.getOrCreate();
	String filePath = "titanic.csv";
	Dataset<Row> df = spark.read() .option("header", "true").csv(filePath);
	
	Dataset<Row> df2 = df.withColumn("Age", functions.col("Age").cast("int"))
		.withColumn("Fare", functions.col("Fare").cast("double"));	// Convertir les colonnes dans de nouveaux types
                        
    	df.printSchema(); // Afficher le schéma du DataFrame

    	df2.printSchema(); // Afficher le schéma du nouveau DataFrame

    	df2.show(10); // Afficher les dix premières lignes du nouveau DataFrame
    	df.show(10);
    	
	df.select("Age", "Sex")
      		.show(20); // Sélectionner les vingt premiers éléments des colonnes "Age" et "Sex"
	
	df.show(10);
	
	    // Afficher les modalités différentes de la colonne "Pclass"
    	df.select("Pclass")
      		.distinct()
      		.show();

    		// Combien de nombres de "parents/enfants à bord" différents y a-t-il dans le jeu de données ?
    	long distinctCount = df.select("Parents/Children Aboard")
                          .distinct()
                          .count();
    	System.out.println("Nombre de nombres de 'parents/enfants à bord' différents : " + distinctCount);

    		// Afficher le descriptif de df2
    	df2.describe().show();

    		// Afficher le nombre de survivants et le nombre de morts
    	df.groupBy("Survived")
     		.count()
     		.withColumn("Survived", functions.when(functions.col("Survived").equalTo(0), "Nombre de mort").otherwise("Nombre de survivant"))
      		.show();
}
}





