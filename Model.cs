namespace MyNeuralNetwork
{
    public class Model
    {
        private static readonly string ProjectDirectory = Path.Combine(Directory.GetParent(Environment.CurrentDirectory)?.Parent?.Parent?.FullName ?? string.Empty, "Data");
        private static readonly Random Rand = new Random();

        public int NetworkID { get; set; }
        public int InputNeurons { get; }
        public int OutputNeurons { get; }
        public int HiddenNeurons { get; }
        public double[,]? HiddenWeights { get; private set; }
        public double[]? HiddenBias { get; private set; }
        public double[,]? OutputWeights { get; private set; }
        public double[]? OutputBias { get; private set; }
        public double NetworkError { get; set; }
        public double NetworkWrongPercentage { get; set; }

        public int Epochs { get; set; }
        public double LearningRate { get; set; }
        public double TotalError { get; set; }
        public double WrongPercentage { get; set; }

        public double[][]? trainingInputs { get; set; }
        public double[][]? trainingOutputs { get; set; }

        public Model(int input, int hidden, int output)
        {
            InputNeurons = input;
            HiddenNeurons = hidden;
            OutputNeurons = output;

            HiddenWeights = GenerateRandomMatrix(input, hidden);
            OutputWeights = GenerateRandomMatrix(hidden, output);
            HiddenBias = GenerateRandomArray(hidden);
            OutputBias = GenerateRandomArray(output);

            Epochs = 1000;
            LearningRate = 0.97;

            trainingInputs = RetrieveInputDataFromCSV($"{ProjectDirectory}\\TrainingData.csv");
            trainingOutputs = RetrieveOutputDataFromCSV($"{ProjectDirectory}\\TrainingData.csv");
        }


        public Model()
        {
        }

        private static double[,] GenerateRandomMatrix(int rows, int cols)
        {
            var matrix = new double[rows, cols];
            for (var i = 0; i < rows; i++)
                for (var j = 0; j < cols; j++)
                    matrix[i, j] = Rand.NextDouble();
            return matrix;
        }

        private static double[] GenerateRandomArray(int size)
        {
            return Enumerable.Range(0, size).Select(_ => Rand.NextDouble()).ToArray();
        }

        // save all weights and biases to csv files
        public void SaveNetworkToCSV()
        {
            SaveMatrixToCSV(HiddenWeights, "HiddenWeights.csv");
            SaveMatrixToCSV(OutputWeights, "OutputWeights.csv");
            SaveArrayToCSV(HiddenBias, "HiddenBias.csv");
            SaveArrayToCSV(OutputBias, "OutputBias.csv");
        }

        // load all weights and biases to csv files.
        public void LoadNetworkFromCSV()
        {
            HiddenWeights = LoadMatrixFromCSV("HiddenWeights.csv", InputNeurons, HiddenNeurons);
            OutputWeights = LoadMatrixFromCSV("OutputWeights.csv", HiddenNeurons, OutputNeurons);
            HiddenBias = LoadArrayFromCSV("HiddenBias.csv");
            OutputBias = LoadArrayFromCSV("OutputBias.csv");
        }

        private void SaveMatrixToCSV(double[,] matrix, string fileName)
        {
            var path = Path.Combine(ProjectDirectory, fileName);
            using var writer = new StreamWriter(path);
            for (var i = 0; i < matrix.GetLength(0); i++)
            {
                var row = Enumerable.Range(0, matrix.GetLength(1)).Select(j => matrix[i, j].ToString());
                writer.WriteLine(string.Join(";", row));
            }
        }

        private void SaveArrayToCSV(double[] array, string fileName)
        {
            var path = Path.Combine(ProjectDirectory, fileName);
            File.WriteAllLines(path, array.Select(x => x.ToString()));
        }

        private double[,] LoadMatrixFromCSV(string fileName, int rows, int cols)
        {
            var path = Path.Combine(ProjectDirectory, fileName);
            var matrix = new double[rows, cols];
            var lines = File.ReadLines(path).ToList();
            for (var i = 0; i < lines.Count; i++)
            {
                var row = lines[i].Split(';').Select(double.Parse).ToArray();
                for (var j = 0; j < row.Length; j++)
                    matrix[i, j] = row[j];
            }
            return matrix;
        }

        private double[] LoadArrayFromCSV(string fileName)
        {
            var path = Path.Combine(ProjectDirectory, fileName);
            return File.ReadLines(path).Select(double.Parse).ToArray();
        }

        public void PrintCurrentWeightsBias()
        {
            PrintMatrix("Hidden Weights", HiddenWeights);
            PrintMatrix("Output Weights", OutputWeights);
            PrintArray("Hidden Bias", HiddenBias);
            PrintArray("Output Bias", OutputBias);
        }

        private static void PrintMatrix(string title, double[,] matrix)
        {
            Console.WriteLine($"\n{title}\n{new string('-', title.Length)}");
            for (var i = 0; i < matrix.GetLength(0); i++)
            {
                var row = Enumerable.Range(0, matrix.GetLength(1)).Select(j => matrix[i, j].ToString());
                Console.WriteLine(string.Join("\t", row));
            }
        }

        private static void PrintArray(string title, double[] array)
        {
            Console.WriteLine($"\n{title}\n{new string('-', title.Length)}");
            Console.WriteLine(string.Join("\t", array));
        }

        private double[][] RetrieveInputDataFromCSV(string filePath)
        {
            string[] lines = File.ReadAllLines(filePath); ;
            double[][] matrix = new double[lines.Length][];

            for (int i = 0; i < lines.Length; i++)
            {
                string[] rowValues = lines[i].Split(';');
                if (rowValues.Length < InputNeurons)
                {
                    Console.WriteLine("Error in Retrieve Data from CSV");
                    return matrix;
                }
                else
                {
                    matrix[i] = rowValues
                    .Take(InputNeurons)
                    .Select(double.Parse)
                    .ToArray();
                }
            }
            return matrix;
        }

        private double[][] RetrieveOutputDataFromCSV(string filePath)
        {
            string[] lines = File.ReadAllLines(filePath); ;
            double[][] matrix = new double[lines.Length][];

            for (int i = 0; i < lines.Length; i++)
            {
                string[] rowValues = lines[i].Split(';');
                // Split each line into an array of strings (columns)
                matrix[i] = rowValues
                    .Skip(Math.Max(0, rowValues.Length - OutputNeurons))
                    .Select(double.Parse)
                    .ToArray();
            }
            return matrix;
        }

        public void TrainModel()
        {
            for (int epoch = 0; epoch < Epochs; epoch++)
            {
                double totalError = 0;
                double CountWrong = 0.0;

                for (int i = 0; i < trainingInputs.Length; i++)
                {
                    // stores the weights of the hiddenlayer inputs
                    double[] hiddenLayerInput = AddBias(MatrixVectorMultiply(HiddenWeights, trainingInputs[i]), HiddenBias);

                    // stores the output of all hiddenlayer neurons
                    double[] hiddenLayerOutput = ApplyActivationFunction(hiddenLayerInput);

                    // stores the weights of the outputlayer inputs
                    double[] outputLayerInput = AddBias(MatrixVectorMultiply(OutputWeights, hiddenLayerOutput), OutputBias);

                    // stores the output of all outputlayer neurons
                    double[] output = ApplyActivationFunction(outputLayerInput);

                    double maxValueIst = output.Max();
                    double IndexIst = output.ToList().IndexOf(maxValueIst);
                    double maxValueSoll = trainingOutputs[i].Max();
                    double IndexSoll = trainingOutputs[i].ToList().IndexOf(maxValueSoll);

                    if (IndexIst != IndexSoll)
                    {
                        CountWrong++;

                        if (epoch == Epochs - 1)
                        {
                            Console.WriteLine("\nFalsch Interpretierter Datensatz:");
                            Console.WriteLine("         \t AA\tAU\tEDB\tGT\tINDL\tINA\tINP\tKV\tKR\tLOG\tPM");
                            Console.Write("IstOutput :\t");
                            for (int j = 0; j <= 10; j++)
                            {
                                if (j == IndexIst)
                                { Console.ForegroundColor = ConsoleColor.Green; }
                                else { Console.ResetColor(); }

                                Console.Write($"{output[j]:F3}\t");
                            }
                            Console.WriteLine("\n");
                            Console.Write("SollOutput:\t");
                            for (int j = 0; j <= 10; j++)
                            {
                                if (j == IndexSoll)
                                { Console.ForegroundColor = ConsoleColor.Green; }
                                else { Console.ResetColor(); }
                                Console.Write($"{trainingOutputs[i][j]:F3}\t");
                            }
                            Console.ResetColor();
                            Console.WriteLine("\n");
                        }
                    }

                    // Fehlerberechnung (Mean Squared Error)
                    double[] errors = new double[output.Length];
                    for (int j = 0; j < output.Length; j++)
                    {
                        errors[j] = trainingOutputs[i][j] - output[j];
                        totalError += errors[j] * errors[j];
                    }

                    // Backpropagation
                    double[] outputDeltas = new double[output.Length];
                    for (int j = 0; j < output.Length; j++)
                    {
                        outputDeltas[j] = errors[j] * output[j] * (1 - output[j]);
                    }

                    double[] hiddenDeltas = new double[hiddenLayerOutput.Length];
                    for (int j = 0; j < hiddenLayerOutput.Length; j++)
                    {
                        hiddenDeltas[j] = 0;
                        for (int k = 0; k < output.Length; k++)
                        {
                            hiddenDeltas[j] += outputDeltas[k] * OutputWeights[j, k];
                        }
                        hiddenDeltas[j] *= hiddenLayerOutput[j] * (1 - hiddenLayerOutput[j]);
                    }

                    // Aktualisierung der Gewichte und Biases im Output-Layer
                    for (int j = 0; j < hiddenLayerOutput.Length; j++)
                    {
                        for (int k = 0; k < output.Length; k++)
                        {
                            OutputWeights[j, k] += LearningRate * outputDeltas[k] * hiddenLayerOutput[j];
                        }
                    }
                    for (int j = 0; j < OutputBias.Length; j++)
                    {
                        OutputBias[j] += LearningRate * outputDeltas[j];
                    }

                    // Aktualisierung der Gewichte und Biases im Hidden-Layer
                    for (int j = 0; j < HiddenWeights.GetLength(0); j++)
                    {
                        for (int k = 0; k < HiddenWeights.GetLength(1); k++)
                        {
                            HiddenWeights[j, k] += LearningRate * hiddenDeltas[k] * trainingInputs[i][j];
                        }
                    }
                    for (int j = 0; j < HiddenBias.Length; j++)
                    {
                        HiddenBias[j] += LearningRate * hiddenDeltas[j];
                    }
                }

                double wrongPercent = CountWrong / trainingInputs.Length * 100.0;
                TotalError = totalError;
                WrongPercentage = wrongPercent;

                // Ausgabe des Fehlers nach jeder 100sten Epoche
                Console.WriteLine($"Epoch {epoch + 1}\t Error: {totalError:F8}\t Fehlerprozent: {wrongPercent:F2}");
            }

        }

        public double[] MatrixVectorMultiply(double[,] matrix, double[] vector)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            double[] result = new double[cols];
            for (int j = 0; j < cols; j++)
            {
                double sum = 0;
                for (int i = 0; i < rows; i++)
                {
                    sum += matrix[i, j] * vector[i];
                }
                result[j] = sum;
            }
            return result;
        }

        public double[] AddBias(double[] vector, double[] bias)
        {
            if (vector.Length != bias.Length)
            {
                throw new ArgumentException("Die Länge des Vektors und der Bias-Dimension stimmen nicht überein.");
            }
            for (int i = 0; i < vector.Length; i++)
            {
                vector[i] += bias[i];
            }
            return vector;
        }

        public double[] ApplyActivationFunction(double[] input)
        {
            double[] output = new double[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                output[i] = 1 / (1 + Math.Exp(-input[i]));
            }
            return output;
        }

        public void CheckSampleData()
        {
            double[][] checkData =
            {
                new double[] { 1, 1, 1, 5.5, 5, 5.5, 0.333}
            };

            double[] hiddenLayerInput = AddBias(MatrixVectorMultiply(HiddenWeights, checkData[0]), HiddenBias);

            // stores the output of all hiddenlayer neurons
            double[] hiddenLayerOutput = ApplyActivationFunction(hiddenLayerInput);

            // stores the weights of the outputlayer inputs
            double[] outputLayerInput = AddBias(MatrixVectorMultiply(OutputWeights, hiddenLayerOutput), OutputBias);

            // stores the output of all outputlayer neurons
            double[] output = ApplyActivationFunction(outputLayerInput);

            Console.WriteLine($"{output[0]:F3},{output[1]:F3},{output[2]:F3},{output[3]:F3}");
        }

    }
}