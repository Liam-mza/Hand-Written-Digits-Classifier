package cs107KNN;

import java.util.Arrays;

public class KNN {
	public static void main(String[] args) {

		int TESTS = 1000;
		int K = 7;
		byte[][][] trainImages = parseIDXimages(Helpers.readBinaryFile("datasets/reduced10Kto1K_images"));
		byte[] trainLabels = parseIDXlabels(Helpers.readBinaryFile("datasets/reduced10Kto1K_labels"));
		byte[][][] testImages = parseIDXimages(Helpers.readBinaryFile("datasets/10k_images_test"));
		byte[] testLabels = parseIDXlabels(Helpers.readBinaryFile("datasets/10k_labels_test"));
		byte[] predictions = new byte[TESTS];
		long start = System.currentTimeMillis();
		for (int i = 0; i < TESTS; i++) {
			predictions[i] = knnClassify(testImages[i], trainImages, trainLabels, K);
		}
		long end = System.currentTimeMillis();
		System.out.println(trainImages.length);
		System.out.println(testImages.length);
		double time = (end - start) / 1000d;
		System.out
		.println("Accuracy = " + accuracy(predictions, Arrays.copyOfRange(testLabels, 0, TESTS)) * 100 + " %");
		System.out.println("Time = " + time + " seconds");
		System.out.println("Time per test image = " + (time / TESTS));
		Helpers.show("Test", testImages, predictions, testLabels, 20, 50);

	}

	/**
	 * Composes four bytes into an integer using big endian convention.
	 *
	 * @param bXToBY The byte containing the bits to store between positions X and Y
	 * 
	 * @return the integer having form [ b31ToB24 | b23ToB16 | b15ToB8 | b7ToB0 ]
	 */
	public static int extractInt(byte b31ToB24, byte b23ToB16, byte b15ToB8, byte b7ToB0) {

		// Shifting des entiers obtenus avec chacun des bytes en fonction de leur ordre

		int n1 = (b7ToB0 & 0xFF);
		int n2 = (b15ToB8 & 0xFF) << 8;
		int n3 = (b23ToB16 & 0xFF) << 16;
		int n4 = (b31ToB24 & 0xFF) << 24;

		// Addition des entiers avec l'opérateur OR		

		int nombre = n1 | n2 | n3 | n4;

		// Retour de l'entier que forme les 4 bytes donnés			

		return nombre;
	}

	/**
	 * Parses an IDX file containing images
	 *
	 * @param data the binary content of the file
	 *
	 * @return A tensor of images
	 */
	public static byte[][][] parseIDXimages(byte[] data) {

		//Détermination du nombre magique du fichier

		int magicNumber = extractInt(data[0], data[1], data[2], data[3]);

		// Vérification que le nombre magique correspond bien à celui d'un fichier d'images

		if (magicNumber != 2051) {
			return null;
		}

		// Détermination du nombre du nombre d'image et de leurs dimensions		

		int nbImages = extractInt(data[4], data[5], data[6], data[7]);
		int nbRows = extractInt(data[8], data[9], data[10], data[11]);
		int nbColums = extractInt(data[12], data[13], data[14], data[15]);

		// Vérification que les dimensions ne soient pas nulles		

		if ((nbColums == 0) || (nbRows == 0) || (nbImages == 0)) {
			return null;
		}

		// Initialisation du tenseur d'images	

		byte[][][] tenseurImg = new byte[nbImages][nbRows][nbColums];

		// Remplissage du tenseur

		int nbPixel = 16; // Ce nombre est initialisé à 16 car les bytes d'indices 0 à 15 ne représentent pas des pixels

		for (int idxImages = 0; idxImages < nbImages; ++idxImages) {

			for (int idxRows = 0; idxRows < nbRows; ++idxRows) {

				for (int idxColums = 0; idxColums < nbColums; ++idxColums) {

					tenseurImg[idxImages][idxRows][idxColums] = (byte) ((data[nbPixel] & 0xFF) - 128);
					++nbPixel;
				}
			}
		}

		// Retour du tenseur contenant les images

		return tenseurImg;
	}

	/**
	 * Parses an idx images containing labels
	 *
	 * @param data the binary content of the file
	 *
	 * @return the parsed labels
	 */
	public static byte[] parseIDXlabels(byte[] data) {

		// Détermination du nombre magique du fichier	

		int magicNumberLbl = extractInt(data[0], data[1], data[2], data[3]);

		// Vérification que le nombre magique correspond bien à celui d'un fichier d'étiquettes		

		if (magicNumberLbl != 2049) {
			return null;
		}

		// Détermination du nombre d'étiquettes		

		int nbLabels = extractInt(data[4], data[5], data[6], data[7]);

		// Vérification que le nombre d'étiquettes ne soit pas nul		

		if (nbLabels == 0) {
			return null;
		}

		// Initialisation du tenseur d'étiquettes

		byte[] tenseurLbl = new byte[nbLabels];

		// Remplissage du tenseur		

		int numeroDuByte = 8; // Ce nombre est initialisé à 8 car les bytes d'indices 0 à 7 ne représentent pas des étiquettes

		for (int idxLabels = 0; idxLabels < nbLabels; ++idxLabels) {
			tenseurLbl[idxLabels] = data[numeroDuByte];
			++numeroDuByte;
		}

		// Retour du tenseur d'étiquettes		

		return tenseurLbl;
	}

	/**
	 * @brief Computes the squared L2 distance of two images
	 * 
	 * @param a, b two images of same dimensions
	 * 
	 * @return the squared euclidean distance between the two images
	 */
	public static float squaredEuclideanDistance(byte[][] a, byte[][] b) {

		float distance = 0;

		// Calcul de la distance entre les deux images

		for (int h = 0; h < a.length; ++h) {
			for (int l = 0; l < a[h].length; ++l) {
				distance += ((a[h][l]) - (b[h][l])) * ((a[h][l]) - (b[h][l]));
			}
		}
		// Retour de la distance

		return distance;
	}

	/**
	 * @brief Computes the inverted similarity between 2 images.
	 * 
	 * @param a, b two images of same dimensions
	 * 
	 * @return the inverted similarity between the two images
	 */
	public static float invertedSimilarity(byte[][] a, byte[][] b) {

		// Calcul des moyennes des images A et B

		float moyA = 0;
		float moyB = 0;

		for (int h = 0; h < b.length; ++h) {
			for (int l = 0; l < b[h].length; ++l) {
				moyB += (b[h][l]);
				moyA += (a[h][l]);
			}
		}
		moyB = moyB / (float) (b.length * b[0].length);
		moyA = moyA / (float) (a.length * a[0].length);

		//Calcul du numérateur dans notre expression
		//Calcul de la première partie de notre dénominateur
		//Calcul de la deuxième partie de notre dénominateur

		float denomA = 0;
		float numerateur = 0;
		float denomB = 0;
		for (int h = 0; h < a.length; ++h) {
			for (int l = 0; l < a[h].length; ++l) {
				numerateur += (a[h][l] - moyA) * (b[h][l] - moyB);
				denomA += (a[h][l] - moyA) * (a[h][l] - moyA);
				denomB += (b[h][l] - moyB) * (b[h][l] - moyB);
			}
		}

		//Calcul du dénominateur

		float denominateur = (float) Math.sqrt(denomA * denomB);
		if (denominateur == 0) {
			return 2;
		}

		//Calcul de notre expression finale de la distance

		float similInv = 1 - (numerateur / denominateur);

		// Retour de la distance

		return similInv;
	}

	/**
	 * @brief Quicksorts and returns the new indices of each value.
	 * 
	 * @param values the values whose indices have to be sorted in non decreasing
	 *               order
	 * 
	 * @return the array of sorted indices
	 * 
	 *         Example: values = quicksortIndices([3, 7, 0, 9]) gives [2, 0, 1, 3]
	 */
	public static int[] quicksortIndices(float[] values) {

		// Initialisation du tableau d'indices	

		int[] indice = new int[values.length];

		// Remplissage du tableau d'indices		

		for (int i = 0; i < indice.length; ++i) {
			indice[i] = i;
		}
		
		// Triage des tableau values et indice en simultané selon l'algoritme de quicksort

		quicksortIndices(values, indice, 0, (values.length - 1));

		// Retour du tableau d'indices trié		

		return indice;
	}

	/**
	 * @brief Sorts the provided values between two indices while applying the same
	 *        transformations to the array of indices
	 * 
	 * @param values  the values to sort
	 * @param indices the indices to sort according to the corresponding values
	 * @param         low, high are the **inclusive** bounds of the portion of array
	 *                to sort
	 */
	public static void quicksortIndices(float[] values, int[] indices, int low, int high) {

		// Triage des valeurs du tableau values selon un ordre croissant avec la technique du quicksort

		/*
		 * Lorsque des valeurs de Values sont permutées de position les valeur de même
		 * indice sont aussi permutées dans le tableau indices afin de garder
		 * l'information relative aux positions initiales de chaque valeur afin de
		 * savoir à quelle image elles correspondent
		 */

		int l = low;
		int h = high;
		float pivot = values[l];
		while (l <= h) {
			if (values[l] < pivot) {
				++l;
			} else if (values[h] > pivot) {
				--h;
			} else {
				swap(l, h, values, indices);
				++l;
				--h;
			}
		}
		if (low < h) {
			quicksortIndices(values, indices, low, h);
		}
		if (high > l) {
			quicksortIndices(values, indices, l, high);
		}
	}

	/**
	 * @brief Swaps the elements of the given arrays at the provided positions
	 * 
	 * @param         i, j the indices of the elements to swap
	 * @param values  the array floats whose values are to be swapped
	 * @param indices the array of ints whose values are to be swapped
	 */
	public static void swap(int i, int j, float[] values, int[] indices) {

		// Permutation de la valeur d'indice i et de la valeur d'indice j dans le tableau values	

		float swap = values[i];
		values[i] = values[j];
		values[j] = swap;

		// Permutation de la valeur d'indice i et de la valeur d'indice j dans le tableau values		

		int swapIndices = indices[i];
		indices[i] = indices[j];
		indices[j] = swapIndices;

	}

	/**
	 * @brief Returns the index of the largest element in the array
	 * 
	 * @param array an array of integers
	 * 
	 * @return the index of the largest integer
	 */
	public static int indexOfMax(int[] array) {

		// Recherche de la plus grande valeur contenue dans le tableau (et de son indice)	

		int indiceOfMax = 0;
		int valueOfMax = array[0];
		for (int j = 0; j < array.length; ++j) {
			if (array[j] > valueOfMax) {
				valueOfMax = array[j];
				indiceOfMax = j;
			}
		}

		// Retour de l'indice de la plus grande valeur du tableau 

		return indiceOfMax;
	}

	/**
	 * The k first elements of the provided array vote for a label
	 *
	 * @param sortedIndices the indices sorted by non-decreasing distance
	 * @param labels        the labels corresponding to the indices
	 * @param k             the number of labels asked to vote
	 *
	 * @return the winner of the election
	 */
	public static byte electLabel(int[] sortedIndices, byte[] labels, int k) {

		byte valeurElue;

		// Création du tableau qui comptabilise le nombre de vote pour chaque étiquette		

		int[] vote = new int[10];

		// Comptabilisation des votes des K images les plus proches de notre image à prédire 		

		for (int i = 0; i < k; ++i) {
			int numeroImage = sortedIndices[i];
			byte valeurLabel = labels[numeroImage];
			++vote[valeurLabel];
		}

		// Recherche du nombre ayant reçu le plus de votes (qui est égal à l'indice de la plus grande valeur du tableau vote)		

		valeurElue = (byte) indexOfMax(vote);

		// Retour du nombre ayant reçu le plus de votes

		return valeurElue;
	}

	/**
	 * Classifies the symbol drawn on the provided image
	 *
	 * @param image       the image to classify
	 * @param trainImages the tensor of training images
	 * @param trainLabels the list of labels corresponding to the training images
	 * @param k           the number of voters in the election process
	 *
	 * @return the label of the image
	 */
	public static byte knnClassify(byte[][] image, byte[][][] trainImages, byte[] trainLabels, int k) {

		// Création du tableau qui va contenir la distance entre notre image à prédire et chaque image de notre set d'entrainement

		float[] tabDistances = new float[trainImages.length];

		// Calcul des distances selon une des 2 méthodes et remplissage du tableau des distances 

		for (int i = 0; i < trainImages.length; ++i) {
			tabDistances[i] = invertedSimilarity(image, trainImages[i]);

			// Pour utiliser la méthode de la distance Euclidienne remplacer la ligne précédente par: 
			//			tabDistances[i]=squaredEuclideanDistance(image, trainImages[i]); 			

		}
		// Initialisation du tableau d'indices		

		int[] sortedIndice = new int[tabDistances.length];

		// Triage grâce à quicksortIndices du tableau des distance pour avoir les indices des images de la plus 
		// proche à la plus éloignée de notre image à prédire 

		sortedIndice = quicksortIndices(tabDistances);

		// Détermination de l'étiquette la plus fréquente parmis les k images les plus proches de notre image à prédire		

		byte valeurElue = electLabel(sortedIndice, trainLabels, k);

		// Retour du nombre prédit		

		return valeurElue;
	}

	/**
	 * Computes accuracy between two arrays of predictions
	 * 
	 * @param predictedLabels the array of labels predicted by the algorithm
	 * @param trueLabels      the array of true labels
	 * 
	 * @return the accuracy of the predictions. Its value is in [0, 1]
	 */
	public static double accuracy(byte[] predictedLabels, byte[] trueLabels) {

		// Calcul de la précision de nos prédictions (retourne un nombre entre 0 et 1 à multiplier par 100 ultérieurement 
		// pour obtenir un pourcentage)	

		double accuracy = 0;
		for (int i = 0; i < predictedLabels.length; ++i) {
			if (predictedLabels[i] == trueLabels[i]) {
				accuracy += 1;
			}
		}
		
		// retour de la précision 

		return (accuracy / trueLabels.length);
	}
}
