/**
 *Nom et prénoms : CHABI ADJOBO AYEDESSO 
 *Cette classe permet de manipuler les Dataset/DataFrame titanic (https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv) 
 	*et penguins_size (https://assets-datascientest.s3-eu-west-1.amazonaws.com/de/resources/datasets/penguins_size.csv)
 * La liste des packages utilisés : import org.apache.spark.sql.SparkSession; import org.apache.spark.sql.Dataset;
 	*import org.apache.spark.sql.Row; import org.apache.spark.sql.functions; 
 	*import org.apache.spark.sql.DataFrameNaFunctions; import java.util.HashMap; import java.util.Map;
 *
 */


import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.DataFrameNaFunctions;
import java.util.HashMap;
import java.util.Map;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StandardScaler;
import org.apache.spark.ml.feature.StandardScalerModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaPairRDD;
import scala.Tuple2;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;



public class titanic {
  public static void main(String[] args) {
	SparkSession spark = SparkSession.builder()
    		.appName("RowDataset")  // Définit le nom de l'application Spark
    		.getOrCreate();
	
	/**
	*DataFrame titanic
	*/
	
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
    	long nbr_parentenfant = df.select("Parents/Children Aboard")
                          .distinct()
                          .count();
    	System.out.println("Nombre de nombres de 'parents/enfants à bord' différents : " + nbr_parentenfant);

    		// Afficher le descriptif de df2
    	df2.describe().show();

    		// Afficher le nombre de survivants et le nombre de morts
    	df.groupBy("Survived")
     		.count()
     		.withColumn("Survived", functions.when(functions.col("Survived").equalTo(0), "Nombre de mort").otherwise("Nombre de survivant"))
      		.show();
      		    // Filtrer les données pour obtenir le nombre de survivants et de morts de moins de 20 ans
    	long survivants_moins_20 = df.filter(functions.col("Survived").equalTo("1").and(functions.col("Age").lt(20))).count();
    	long morts_moins_20 = df.filter(functions.col("Survived").equalTo("0").and(functions.col("Age").lt(20))).count();
    	
    	df.filter(functions.col("Age").lt(20)).groupBy("Survived").count().show();

    	System.out.println("Nombre de survivants de moins de 20 ans : " + survivants_moins_20);
    	System.out.println("Nombre de morts de moins de 20 ans : " + morts_moins_20);
    	
    	
    	/*
    	*Dataframe penguins_size
    	*/
    	
    	 	// Charger le fichier "penguins_size.csv" dans le DataFrame df_penguins
    	String penguinsFilePath = "penguins_size.csv";
    	Dataset<Row> df_penguins = spark.read().option("header", "true").csv(penguinsFilePath);

    		// Afficher les 10 premières lignes du DataFrame df_penguins
    	df_penguins.show(10);

    		// Afficher le schéma du DataFrame df_penguins
    	df_penguins.printSchema();
    	
    		// Afficher les valeurs manquantes
    	Dataset<Row> df_penguins2 = df_penguins.withColumn("culmen_length_mm", functions.col("culmen_length_mm").cast("float"))
    		.withColumn("culmen_depth_mm", functions.col("culmen_depth_mm").cast("float"))
    		.withColumn("flipper_length_mm", functions.col("flipper_length_mm").cast("float"))
    		.withColumn("body_mass_g", functions.col("body_mass_g").cast("float"));
    	
    	String[] col_names = df_penguins2.columns();
	int num_cols = col_names.length;
	Row RR = df_penguins2.describe().drop("summary").head();
	for (int i = 0; i < num_cols; i++) {
  		System.out.println(col_names[i] + ": " + (df_penguins2.count()-Integer.parseInt(RR.getString(i))));
    }
    	
   	String[] categoricalColumns = {"species", "island", "sex"};
    	for (String column : categoricalColumns) {
        	long distinctCount = df_penguins2.select(column).distinct().count();
        	System.out.println("Nombre de modalités pour la colonne " + column + " : " + distinctCount);
	  	
}
		// Afficher le nombre de modalités de la colonne "species"
	df_penguins2.groupBy("species").count().show();

		// Afficher le nombre de modalités de la colonne "island"
	df_penguins2.groupBy("island").count().show();

		// Afficher le nombre de modalités de la colonne "sex"
	df_penguins2.groupBy("sex").count().show();
	
	
	
	String[] str1 = {"culmen_length_mm"};
	String[] str2 = {"culmen_depth_mm"};
	String[] str3 = {"flipper_length_mm"};
	String[] str4 = {"body_mass_g"};
	
		// Remplacer les valeurs manquantes dans un nouveau DataFrame
	Dataset<Row> df_penguins3 = df_penguins2
		.na().fill(43.922, str1)
		.na().fill(17.151, str2)
		.na().fill(200.915, str3)
		.na().fill(4201.754, str4);

		// Afficher le descriptif du DataFrame df_penguins3
	df_penguins3.describe().show();

		// Afficher la modalité la plus fréquente de la colonne 'sex'
	String modalite_frequente = df_penguins3.groupBy("sex").count().orderBy(functions.desc("count")).first().getString(0);
	System.out.println("Modalité la plus fréquente de la colonne 'sex' : " + modalite_frequente);

		// Remplacer les valeurs manquantes dans la colonne 'sex'
	Map<String, String> sex_replace = new HashMap<>();
	sex_replace.put("NA", modalite_frequente);
	sex_replace.put(".", modalite_frequente);
	df_penguins3 = df_penguins3.na().replace("sex", sex_replace);

		// Afficher le descriptif du DataFrame df_penguins3
	df_penguins3.describe().show();


			// Remplacer les valeurs manquantes des autres variables catégorielles par leur modalités fréquentes
	String[] categoricalColumns2 = {"species", "island"};
	for (String column : categoricalColumns2) {
		    String frequent_modalite = df_penguins3.groupBy(column).count().orderBy(functions.desc("count")).first().getString(0);
		    Map<String, String> replace_map = new HashMap<>();
		    replace_map.put("NA", frequent_modalite);
		    replace_map.put(".", frequent_modalite);
		    df_penguins3 = df_penguins3.na().replace(column, replace_map);
}

		// Vérifier que les colonnes catégorielles ne contiennent plus de valeurs manquantes
	String[] categoricalColumns3 = {"species", "island", "sex"};
	for (String column : categoricalColumns3) {
	    long missingCount = df_penguins3.filter(df_penguins3.col(column).isNull().or(df_penguins3.col(column).equalTo(""))).count();
	    System.out.println("Nombre de valeurs manquantes pour la colonne " + column + " : " + missingCount);
	}




	/*
	Machine leaaring sur big data
	*/
	
	/*
	1- Pre-processing des données : 
	Les deux transformations que nous allons appliquer à nos données sont l'indexation des variables catégorielle (en labels) 
	et la standardisation des variables numériques (la réduction des données).
	*/
	 /* 1-1 indexation des variables catégorielle (en labels) */
	 
	Dataset<Row> df_raw = df_penguins3;
	df_raw.show(20);

	// Création d'une instance de StringIndexer
	StringIndexer indexer = new StringIndexer()
		.setInputCols(new String[] {"species", "island", "sex"})
		.setOutputCols(new String[] {"speciesIndex", "islandIndex", "sexIndex"});
		
	// Application de la transformation sur un DataFrame
	Dataset<Row> indexed = indexer.fit(df_raw).transform(df_raw);
	indexed.show();
	indexed.describe().show();
	
	/* 1-2 standardisation des variables numériques (la réduction des données)*/
	
	VectorAssembler assembler = new VectorAssembler()
		.setInputCols(new String[] {"culmen_length_mm", "culmen_depth_mm","flipper_length_mm", "body_mass_g"})
		.setOutputCol("features");
	Dataset<Row> assembled = assembler.transform(indexed);
	assembled.select("features").show();
	
	
		// Entraînement du modèle de StandardScaler
		//StandardScalerModel assembler_f = assembler.fit(df_raw)
	StandardScaler scaler = new StandardScaler()
    		.setInputCol("features")
    		.setOutputCol("scaledFeatures")
    		.setWithStd(true)
    		.setWithMean(true);




		// Transformation du DataFrame en utilisant le modèle entraîné
	StandardScalerModel scalerModel = scaler.fit(assembled);
	Dataset<Row> scaled = scalerModel.transform(assembled);
	scaled.show();
	scaled.describe().show();
	
	
	VectorAssembler assembler_fin = new VectorAssembler()
		.setInputCols(new String[] {"scaledFeatures", "islandIndex", "sexIndex"})
		.setOutputCol("big_features");
	Dataset<Row> data = assembler_fin
		.transform(scaled)
		.select("speciesIndex", "big_features");
	data = data.withColumnRenamed("speciesIndex", "label");
	data = data.withColumnRenamed("big_features", "features");
	data.show();


/*
	Modèles et Pipeline 
	*/
		// Train a DecisionTree model.
  	LogisticRegression lr = new LogisticRegression()
		.setLabelCol("label")
		.setFeaturesCol("features");
  	
	LogisticRegressionModel lrModel = lr.fit(data);


	/*JavaRDD<Row> data_rdd = data.toJavaRDD();
	JavaPairRDD<Object, Object> predictionAndLabels = data_rdd.mapToPair(p ->
	  new Tuple2<>(lrModel.predict(p.getAs(1)), p.getAs(0)));
	MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels.rdd());
	System.out.format("Weighted precision = %f\n", metrics.weightedPrecision());*/
	
	
	

		// mettre la base de donnée au format svmlib en appliquant un StringIndexer aux variables catégorielles.
	StringIndexer indexer_pre = new StringIndexer()
	    	.setInputCols(new String[] {"species", "island", "sex"})
	    	.setOutputCols(new String[] {"label", "islandIndex", "sexIndex"});
	Dataset<Row> indexed_pre = indexer_pre.fit(df_raw).transform(df_raw);


	
	VectorAssembler assembler_pre = new VectorAssembler()
	    .setInputCols(new String[] {"islandIndex", "culmen_length_mm", "culmen_depth_mm", "flipper_length_mm", "body_mass_g", "sexIndex"})
	    .setOutputCol("features");
	Dataset<Row> data_pre = assembler_pre.transform(indexed_pre).select("label", "features");
	data_pre.show();
	

/*
	scaler = new StandardScaler()
    		.setInputCol("features2")
    		.setOutputCol("scaledFeatures")
    		.setWithStd(true)
    		.setWithMean(true);
*/

		// Création de la Pipeline
	Pipeline pipeline = new Pipeline().setStages(new PipelineStage[] {scaler, lr});

		// Séparation des données en ensembles d'entraînement et de test
	Dataset<Row>[] data_split = data_pre.randomSplit(new double[]{0.8, 0.2}, 12345);
	Dataset<Row> trainData = data_split[0];
	Dataset<Row> testData = data_split[1];
	

		// Entraînement du modèle sur l'ensemble d'entraînement
	PipelineModel model = pipeline.fit(trainData);

		// Test du modèle sur l'ensemble de test
	Dataset<Row> predictions = model.transform(testData);

	predictions.show();

	JavaRDD<Row> predictions_rdd = predictions.toJavaRDD();
	JavaPairRDD< Object, Object> predictionAndLabels = predictions_rdd.mapToPair(p ->
  	new Tuple2<>(p.getAs(5), p.getAs(0)));
	MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels.rdd());
	System.out.format("Weighted precision = %f\n", metrics.weightedPrecision());
	// Entraînement du modèle sur l'ensemble d'entraînement
	PipelineModel model2 = pipeline.fit(trainData);

	// Test du modèle sur l'ensemble de test
	Dataset<Row> predictions2 = model2.transform(testData);
	predictions2.show();

	JavaRDD<Row> predictions_rdd2 = predictions2.toJavaRDD();
	JavaPairRDD<Object, Object> predictionAndLabels2 = predictions_rdd2.mapToPair(p ->
	    new Tuple2<>(p.getAs("prediction"), p.getAs("label")));
	MulticlassMetrics metrics2 = new MulticlassMetrics(predictionAndLabels2.rdd());
	System.out.format("Weighted precision = %f\n", metrics2.weightedPrecision());

	
}
}





