package com.ontariotechu.sofe3980U;

import java.io.FileReader;
import java.util.List;
import java.util.ArrayList;
import com.opencsv.*;

public class App {
	public static void main(String[] args) {
		// List of file paths for models
		String[] filePaths = {"model_1.csv", "model_2.csv", "model_3.csv"};

		// Track best model performance
		double bestBCE = Double.MAX_VALUE;
		double bestAccuracy = 0.0;
		double bestPrecision = 0.0;
		double bestRecall = 0.0;
		double bestF1Score = 0.0;
		double bestAUC = 0.0;
		String bestModelBCE = "";
		String bestModelAccuracy = "";
		String bestModelPrecision = "";
		String bestModelRecall = "";
		String bestModelF1Score = "";
		String bestModelAUC = "";

		for (String filePath : filePaths) {
			System.out.println("Metrics for file: " + filePath);

			FileReader filereader;
			List<String[]> allData;

			try {
				filereader = new FileReader(filePath);
				CSVReader csvReader = new CSVReaderBuilder(filereader).withSkipLines(1).build();
				allData = csvReader.readAll();
			} catch (Exception e) {
				System.out.println("Error reading the CSV file: " + filePath);
				continue;
			}

			// Initialize variables
			int TP = 0, FP = 0, TN = 0, FN = 0; // Confusion matrix components
			double totalBCE = 0.0; // For Binary Cross-Entropy
			List<Double> predictedProbs = new ArrayList<>();
			List<Integer> actualValues = new ArrayList<>();

			// Loop through the data to populate confusion matrix and calculate BCE
			for (String[] row : allData) {
				int y_true = Integer.parseInt(row[0]);
				float y_predicted = Float.parseFloat(row[1]);
				predictedProbs.add((double) y_predicted); // Predicted probabilities
				actualValues.add(y_true); // Actual values

				// Clip y_predicted to avoid log(0) or log(1)
				double epsilon = 1e-15;
				double y_prob = Math.max(epsilon, Math.min(1 - epsilon, y_predicted));

				// Calculate Binary Cross-Entropy for the current sample
				if (y_true == 1) {
					totalBCE += -Math.log(y_prob);
				} else {
					totalBCE += -Math.log(1 - y_prob);
				}

				// Update confusion matrix
				if (y_true == 1 && y_predicted >= 0.5) TP++; // True Positive
				if (y_true == 1 && y_predicted < 0.5) FN++; // False Negative
				if (y_true == 0 && y_predicted >= 0.5) FP++; // False Positive
				if (y_true == 0 && y_predicted < 0.5) TN++; // True Negative
			}

			// Calculate BCE (average)
			double BCE = totalBCE / allData.size();
			System.out.printf("BCE = %.7f\n", BCE);

			// Calculate metrics based on confusion matrix
			double accuracy = (double) (TP + TN) / (TP + TN + FP + FN);
			double precision = (double) TP / (TP + FP);
			double recall = (double) TP / (TP + FN);
			double f1Score = 2 * (precision * recall) / (precision + recall);

			// Print confusion matrix
			System.out.println("Confusion matrix");
			System.out.println("\ty=1\ty=0");
			System.out.println("y^=1\t" + TP + "\t" + FP);
			System.out.println("y^=0\t" + FN + "\t" + TN);
			System.out.printf("Accuracy = %.4f\n", accuracy);
			System.out.printf("Precision = %.7f\n", precision);
			System.out.printf("Recall = %.8f\n", recall);
			System.out.printf("F1 score = %.8f\n", f1Score);

			// Calculate AUC-ROC
			double auc = calculateAUCROC(predictedProbs, actualValues);
			System.out.printf("AUC ROC = %.8f\n", auc);
			System.out.println("------------------------------------");

			// Track the best performing model based on each metric
			if (BCE < bestBCE) {
				bestBCE = BCE;
				bestModelBCE = filePath;
			}
			if (accuracy > bestAccuracy) {
				bestAccuracy = accuracy;
				bestModelAccuracy = filePath;
			}
			if (precision > bestPrecision) {
				bestPrecision = precision;
				bestModelPrecision = filePath;
			}
			if (recall > bestRecall) {
				bestRecall = recall;
				bestModelRecall = filePath;
			}
			if (f1Score > bestF1Score) {
				bestF1Score = f1Score;
				bestModelF1Score = filePath;
			}
			if (auc > bestAUC) {
				bestAUC = auc;
				bestModelAUC = filePath;
			}
		}

		// Output the model with better performance for each metric
		System.out.println("According to BCE, The best model is " + bestModelBCE);
		System.out.println("According to Accuracy, The best model is " + bestModelAccuracy);
		System.out.println("According to Precision, The best model is " + bestModelPrecision);
		System.out.println("According to Recall, The best model is " + bestModelRecall);
		System.out.println("According to F1 score, The best model is " + bestModelF1Score);
		System.out.println("According to AUC ROC, The best model is " + bestModelAUC);
	}

	/**
	 * Calculate AUC-ROC
	 */
	public static double calculateAUCROC(List<Double> predictedProbs, List<Integer> actualValues) {
		// Sort predicted probabilities and actual values together based on predicted probabilities
		List<Integer> sortedIndices = new ArrayList<>();
		for (int i = 0; i < predictedProbs.size(); i++) {
			sortedIndices.add(i);
		}
		sortedIndices.sort((i, j) -> Double.compare(predictedProbs.get(j), predictedProbs.get(i)));

		// Calculate True Positive Rate (TPR) and False Positive Rate (FPR)
		double auc = 0.0;
		double prevFPR = 0.0, prevTPR = 0.0;
		double totalPositive = 0, totalNegative = 0;
		for (int i = 0; i < actualValues.size(); i++) {
			if (actualValues.get(i) == 1) {
				totalPositive++;
			} else {
				totalNegative++;
			}
		}

		int TP = 0, FP = 0;
		for (int i = 0; i < sortedIndices.size(); i++) {
			int index = sortedIndices.get(i);
			if (actualValues.get(index) == 1) {
				TP++;
			} else {
				FP++;
			}

			double TPR = (double) TP / totalPositive;
			double FPR = (double) FP / totalNegative;
			auc += (TPR + prevTPR) * (FPR - prevFPR) / 2.0;

			prevFPR = FPR;
			prevTPR = TPR;
		}

		return auc;
	}
}
