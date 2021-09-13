package cs107KNN;

import java.util.Set;
import java.util.HashSet;
import java.util.Random;
import java.util.ArrayList;

public class KMeansClustering {
	public static void main(String[] args) {

		int K = 1000;
		int maxIters = 20;

		// Veuillez adapter les parcours en fonction du fichier que vous souhaitez réduire 
		byte[][][] images = KNN.parseIDXimages(Helpers.readBinaryFile("datasets/1000-per-digit_images_train"));
		byte[] labels = KNN.parseIDXlabels(Helpers.readBinaryFile("datasets/1000-per-digit_labels_train"));

		byte[][][] reducedImages = KMeansReduce(images, K, maxIters);

		byte[] reducedLabels = new byte[reducedImages.length];
		for (int i = 0; i < reducedLabels.length; i++) {
			reducedLabels[i] = KNN.knnClassify(reducedImages[i], images, labels, 5);
			System.out.println("Classified " + (i + 1) + " / " + reducedImages.length);
		}

		Helpers.writeBinaryFile("datasets/reduced10Kto1K_images", encodeIDXimages(reducedImages));
		Helpers.writeBinaryFile("datasets/reduced10Kto1K_labels", encodeIDXlabels(reducedLabels));
	}

	/**
	 * @brief Encodes a tensor of images into an array of data ready to be written
	 *        on a file
	 * 
	 * @param images the tensor of image to encode
	 * 
	 * @return the array of byte ready to be written to an IDX file
	 */
	public static byte[] encodeIDXimages(byte[][][] images) {

		// Création du tableau qui contiendra la suite de bytes à écrire

		byte[] imagesInBytes = new byte[images.length * images[0].length * images[0][0].length + 16];

		// Insertion dans le tableau du nombre magique pour les images, du nombre d'images et de leurs dimensions 

		encodeInt(2051, imagesInBytes, 0);
		encodeInt(images.length, imagesInBytes, 4);
		encodeInt(images[0].length, imagesInBytes, 8);
		encodeInt(images[0][0].length, imagesInBytes, 12);

		// Remplissage du tableaau avec les valeurs des pixels 

		int indice = 16;
		for (int i = 0; i < images.length; ++i) {
			for (int j = 0; j < images[0].length; ++j) {
				for (int k = 0; k < images[0][0].length; ++k) {
					imagesInBytes[indice] = (byte) ((images[i][j][k] & 0xFF) + 128);
					++indice;
				}
			}
		}
		return imagesInBytes;
	}

	/**
	 * @brief Prepares the array of labels to be written on a binary file
	 * 
	 * @param labels the array of labels to encode
	 * 
	 * @return the array of bytes ready to be written to an IDX file
	 */
	public static byte[] encodeIDXlabels(byte[] labels) {

		// Création du tableau qui contiendra la suite de bytes à écrire

		byte[] labelsInBytes = new byte[labels.length + 8];

		// Insertion dans le tableau du nombre magique pour les étiquettes, du nombre d'étiquettes		

		encodeInt(2049, labelsInBytes, 0);
		encodeInt(labels.length, labelsInBytes, 4);

		// Remplissage du tableau avec les valeurs des étiquettes	

		int indice = 8;
		for (int i = 0; i < labels.length; ++i) {
			labelsInBytes[indice] = labels[i];
			++indice;
		}
		return labelsInBytes;
	}

	/**
	 * @brief Decomposes an integer into 4 bytes stored consecutively in the
	 *        destination array starting at position offset
	 * 
	 * @param n           the integer number to encode
	 * @param destination the array where to write the encoded int
	 * @param offset      the position where to store the most significant byte of
	 *                    the integer, the others will follow at offset + 1, offset
	 *                    + 2, offset + 3
	 */
	public static void encodeInt(int n, byte[] destination, int offset) {

		// Conversion de l'entier en 4 bytes

		byte b1 = (byte) (n & 0xFF);
		byte b2 = (byte) (n >> 8 & 0xFF);
		byte b3 = (byte) (n >> 16 & 0xFF);
		byte b4 = (byte) (n >> 24 & 0xFF);

		// Remplis des cases d'indice offset à offset+3 du tableau destination avec les 4 bytes calculés
		
		destination[offset] = b4;
		destination[offset + 1] = b3;
		destination[offset + 2] = b2;
		destination[offset + 3] = b1;
	}

	/**
	 * @brief Runs the KMeans algorithm on the provided tensor to return size
	 *        elements.
	 * 
	 * @param tensor   the tensor of images to reduce
	 * @param size     the number of images in the reduced dataset
	 * @param maxIters the number of iterations of the KMeans algorithm to perform
	 * 
	 * @return the tensor containing the reduced dataset
	 */
	public static byte[][][] KMeansReduce(byte[][][] tensor, int size, int maxIters) {
		int[] assignments = new Random().ints(tensor.length, 0, size).toArray();
		byte[][][] centroids = new byte[size][][];
		initialize(tensor, assignments, centroids);

		int nIter = 0;
		while (nIter < maxIters) {
			// Step 1: Assign points to closest centroid
			recomputeAssignments(tensor, centroids, assignments);
			System.out.println("Recomputed assignments");
			// Step 2: Recompute centroids as average of points
			recomputeCentroids(tensor, centroids, assignments);
			System.out.println("Recomputed centroids");

			System.out.println("KMeans completed iteration " + (nIter + 1) + " / " + maxIters);

			nIter++;
		}

		return centroids;
	}

	/**
	 * @brief Assigns each image to the cluster whose centroid is the closest. It
	 *        modifies.
	 * 
	 * @param tensor      the tensor of images to cluster
	 * @param centroids   the tensor of centroids that represent the cluster of
	 *                    images
	 * @param assignments the vector indicating to what cluster each image belongs
	 *                    to. if j is at position i, then image i belongs to cluster
	 *                    j
	 */
	public static void recomputeAssignments(byte[][][] tensor, byte[][][] centroids, int[] assignments) {

		// Création du tableau qui regroupera les distances entre une image et les différents représentants des clusters

		float[] distances = new float[centroids.length];

		// Calcul de la distance entre une image et les différents représentants des clusters et remplissage de assignments		

		for (int i = 0; i < tensor.length; ++i) {
			for (int j = 0; j < centroids.length; ++j) {
				distances[j] = KNN.squaredEuclideanDistance(tensor[i], centroids[j]);

			}
			assignments[i] = IndiceOfMin(distances);
		}
	}

	public static int IndiceOfMin(float[] tab) {
		
// Recherche de l'indice de la valeur minimale d'un tableau
		
		float valMin = tab[0];
		int indiceOfMin = 0;

		for (int i = 0; i < tab.length; ++i) {
			if (tab[i] < valMin) {
				valMin = tab[i];
				indiceOfMin = i;
			}
		}

		return indiceOfMin;
	}

	/**
	 * @brief Computes the centroid of each cluster by averaging the images in the
	 *        cluster
	 * 
	 * @param tensor      the tensor of images to cluster
	 * @param centroids   the tensor of centroids that represent the cluster of
	 *                    images
	 * @param assignments the vector indicating to what cluster each image belongs
	 *                    to. if j is at position i, then image i belongs to cluster
	 *                    j
	 */
	public static void recomputeCentroids(byte[][][] tensor, byte[][][] centroids, int[] assignments) {

		// Initialisation d'un tableau associant chaque case à un cluster

		int[] count = new int[centroids.length];

		// Initialisation d'un tableau calculant la somme de chaque pixel des images, dans chaque different cluster

		float[][][] sum = new float[centroids.length][tensor[0].length][tensor[0][0].length];

		// Itération sur chaque image pour les associer à leurs clusters respectifs, et remplissage du tableau de sommes	

		for (int i = 0; i < tensor.length; ++i) {
			int cluster = assignments[i];
			++count[cluster];
			byte[][] image = tensor[i];
			for (int j = 0; j < image.length; ++j) {
				for (int k = 0; k < image[0].length; ++k) {
					sum[cluster][j][k] += image[j][k];
				}
			}
		}

		// Réinitialisation des centroids en calculant la moyenne de chaque pixel dans chaque cluster, ces pixels "moyens" vont définir nos nouveaux centroids

		for (int i = 0; i < centroids.length; ++i) {
			for (int j = 0; j < centroids[0].length; ++j) {
				for (int k = 0; k < centroids[0][0].length; ++k) {
					centroids[i][j][k] = (byte) (sum[i][j][k] / count[i]);
				}
			}
		}
	}

	/**
	 * Initializes the centroids and assignments for the algorithm. The assignments
	 * are initialized randomly and the centroids are initialized by randomly
	 * choosing images in the tensor.
	 * 
	 * @param tensor      the tensor of images to cluster
	 * @param assignments the vector indicating to what cluster each image belongs
	 *                    to.
	 * @param centroids   the tensor of centroids that represent the cluster of
	 *                    images if j is at position i, then image i belongs to
	 *                    cluster j
	 */
	public static void initialize(byte[][][] tensor, int[] assignments, byte[][][] centroids) {
		Set<Integer> centroidIds = new HashSet<>();
		Random r = new Random("cs107-2018".hashCode());
		while (centroidIds.size() != centroids.length)
			centroidIds.add(r.nextInt(tensor.length));
		Integer[] cids = centroidIds.toArray(new Integer[] {});
		for (int i = 0; i < centroids.length; i++)
			centroids[i] = tensor[cids[i]];
		for (int i = 0; i < assignments.length; i++)
			assignments[i] = cids[r.nextInt(cids.length)];
	}
}
