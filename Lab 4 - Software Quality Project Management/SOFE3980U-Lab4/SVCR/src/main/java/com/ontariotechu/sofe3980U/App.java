package com.ontariotechu.sofe3980U;

import java.io.FileReader;
import java.io.FileNotFoundException;
import java.util.List;
import com.opencsv.*;

public class App
{
	public static void main(String[] args)
	{
		// Store calculated metrics for comparison
		double[] mse = new double[3];
		double[] mae = new double[3];
		double[] mare = new double[3];

		// Evaluate metrics for all three model files
		mse[0] = evaluateFile("model_1.csv", "model_1", "MSE");
		mae[0] = evaluateFile("model_1.csv", "model_1", "MAE");
		mare[0] = evaluateFile("model_1.csv", "model_1", "MARE");

		mse[1] = evaluateFile("model_2.csv", "model_2", "MSE");
		mae[1] = evaluateFile("model_2.csv", "model_2", "MAE");
		mare[1] = evaluateFile("model_2.csv", "model_2", "MARE");

		mse[2] = evaluateFile("model_3.csv", "model_3", "MSE");
		mae[2] = evaluateFile("model_3.csv", "model_3", "MAE");
		mare[2] = evaluateFile("model_3.csv", "model_3", "MARE");

		// Identify the best model for each metric
		System.out.println("\n--- Summary ---");
		System.out.println("According to MSE, The best model is " + getBestModel(mse, "MSE"));
		System.out.println("According to MAE, The best model is " + getBestModel(mae, "MAE"));
		System.out.println("According to MARE, The best model is " + getBestModel(mare, "MARE"));
	}

	/**
	 * Reads the CSV file, calculates the specified metric, and prints the results.
	 */
	public static double evaluateFile(String filePath, String modelName, String metricType)
	{
		FileReader filereader;
		List<String[]> allData;

		try {
			filereader = new FileReader(filePath);
			CSVReader csvReader = new CSVReaderBuilder(filereader).withSkipLines(1).build(); // Skip header row
			allData = csvReader.readAll();

			if (allData.isEmpty()) {
				System.out.println("The file " + filePath + " is empty or only contains a header.");
				return Double.MAX_VALUE;
			}
		} catch (FileNotFoundException e) {
			System.out.println("File not found: " + filePath);
			e.printStackTrace();
			return Double.MAX_VALUE;
		} catch (Exception e) {
			System.out.println("Error reading the CSV file: " + filePath);
			e.printStackTrace();
			return Double.MAX_VALUE;
		}

		int n = allData.size();
		double sumSquaredError = 0.0;
		double sumAbsoluteError = 0.0;
		double sumRelativeError = 0.0;
		double epsilon = 1e-8;  // Small constant to avoid division by zero

		// Process each row in the CSV file
		for (String[] row : allData) {
			try {
				double y_true = Double.parseDouble(row[0]);
				double y_predicted = Double.parseDouble(row[1]);

				double error = y_true - y_predicted;
				double absError = Math.abs(error);

				sumSquaredError += error * error;
				sumAbsoluteError += absError;
				sumRelativeError += absError / (Math.abs(y_true) + epsilon);
			} catch (NumberFormatException e) {
				System.out.println("Skipping invalid data in file " + filePath + ": " + row[0] + ", " + row[1]);
			}
		}

		// Calculate and print the required metric
		double result;
		switch (metricType) {
			case "MSE":
				result = sumSquaredError / n;
				break;
			case "MAE":
				result = sumAbsoluteError / n;
				break;
			case "MARE":
				result = sumRelativeError / n;  // FIXED: Removed unnecessary multiplication by 100
				break;
			default:
				result = Double.MAX_VALUE;
				break;
		}

		System.out.println("For " + modelName);
		System.out.printf("    %s = %.5f\n", metricType, result);
		return result;
	}

	/**
	 * Determines the best model based on the provided metric values.
	 */
	public static String getBestModel(double[] values, String metric) {
		int bestIndex = 0;
		for (int i = 1; i < values.length; i++) {
			if (values[i] < values[bestIndex]) {
				bestIndex = i;
			}
		}
		return "model_" + (bestIndex + 1) + ".csv";
	}
}
