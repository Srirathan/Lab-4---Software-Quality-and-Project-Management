package com.ontariotechu.sofe3980U;

import java.io.FileReader;
import java.util.List;
import com.opencsv.*;
import org.apache.commons.math3.util.Precision;

/**
 * Evaluate Multiclass Classification
 */
public class App {
	public static void main(String[] args) {
		String filePath = "model.csv"; // Path to the model CSV file
		FileReader filereader;
		List<String[]> allData;
		int n = 0; // Total number of samples
		int[][] confusionMatrix = new int[5][5]; // Confusion matrix for 5 classes
		double ce = 0.0; // Cross-entropy
		int numClasses = 5; // Number of classes

		try {
			filereader = new FileReader(filePath);
			CSVReader csvReader = new CSVReaderBuilder(filereader).withSkipLines(1).build();
			allData = csvReader.readAll();
		} catch (Exception e) {
			System.out.println("Error reading the CSV file");
			return;
		}

		// Loop through each row to calculate metrics
		for (String[] row : allData) {
			int trueClass = Integer.parseInt(row[0]); // Actual class label
			double[] predictedProbs = new double[numClasses]; // Predicted probabilities for each class

			// Read the predicted probabilities for each class
			for (int i = 1; i <= numClasses; i++) {
				predictedProbs[i - 1] = Double.parseDouble(row[i]);
			}

			// Calculate Cross-Entropy
			ce -= Math.log(predictedProbs[trueClass - 1]);

			// Find the predicted class (class with the highest probability)
			int predictedClass = 0;
			double maxProb = predictedProbs[0];
			for (int i = 1; i < numClasses; i++) {
				if (predictedProbs[i] > maxProb) {
					maxProb = predictedProbs[i];
					predictedClass = i;
				}
			}

			// Update confusion matrix
			confusionMatrix[trueClass - 1][predictedClass]++;

			n++; // Increment number of samples processed
		}

		// Calculate and print metrics for each class
		double accuracy = 0.0;
		double macroPrecision = 0.0;
		double macroRecall = 0.0;
		double macroF1Score = 0.0;

		for (int i = 0; i < numClasses; i++) {
			int tp = confusionMatrix[i][i]; // True positives for class i
			int fp = 0; // False positives for class i
			int fn = 0; // False negatives for class i

			for (int j = 0; j < numClasses; j++) {
				if (j != i) {
					fp += confusionMatrix[j][i]; // Sum of other rows in the same column
					fn += confusionMatrix[i][j]; // Sum of other columns in the same row
				}
			}

			double precision = (tp + fp == 0) ? 0 : tp / (double) (tp + fp);
			double recall = (tp + fn == 0) ? 0 : tp / (double) (tp + fn);
			double f1Score = (precision + recall == 0) ? 0 : 2 * precision * recall / (precision + recall);

			System.out.println("Class " + (i + 1) + " Metrics:");
			System.out.println("Precision: " + Precision.round(precision, 4));
			System.out.println("Recall: " + Precision.round(recall, 4));
			System.out.println("F1 Score: " + Precision.round(f1Score, 4));

			macroPrecision += precision;
			macroRecall += recall;
			macroF1Score += f1Score;
		}

		// Calculate Accuracy
		int totalCorrect = 0;
		for (int i = 0; i < numClasses; i++) {
			totalCorrect += confusionMatrix[i][i];
		}
		accuracy = (double) totalCorrect / n;
		System.out.println("Accuracy: " + Precision.round(accuracy, 4));

		// Calculate Cross-Entropy
		System.out.println("Cross-Entropy: " + Precision.round(ce / n, 4));

		// Print Confusion Matrix with cleaner alignment
		System.out.println("\nConfusion Matrix:");
		System.out.println("       y=1   y=2   y=3   y=4   y=5");
		for (int i = 0; i < numClasses; i++) {
			System.out.printf("y^=%d   ", i + 1); // Print the row label (predicted class)
			for (int j = 0; j < numClasses; j++) {
				System.out.printf("%-6d", confusionMatrix[i][j]); // Print each cell with left alignment
			}
			System.out.println(); // Move to the next line after each row
		}

		// Calculate Macro Metrics
		System.out.println("Macro Precision: " + Precision.round(macroPrecision / numClasses, 4));
		System.out.println("Macro Recall: " + Precision.round(macroRecall / numClasses, 4));
		System.out.println("Macro F1 Score: " + Precision.round(macroF1Score / numClasses, 4));
	}
}