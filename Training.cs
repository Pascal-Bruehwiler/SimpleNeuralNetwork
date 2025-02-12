namespace MyNeuralNetwork
{
    public class Training
    {
        public static readonly string projectDirectory = Directory.GetParent(Environment.CurrentDirectory).Parent.Parent.FullName;

        private Network _network;
        private int _epochs { get; set; }
        public double _learningRate;
        public double _totalError;
        public double _wrongPercentage;
        public double[][] trainingInputs;
        public double[][] trainingOutputs;

        public Training(Network network)
        {
            _network = network;
            _epochs = 10000;
            _learningRate = 0.05;
            trainingInputs = RetrieveInputDataFromCSV($"{projectDirectory}\\Data\\TrainingData.csv");
            trainingOutputs = RetrieveOutputDataFromCSV($"{projectDirectory}\\Data\\TrainingData.csv");
        }

        private double[][] RetrieveInputDataFromCSV(string filePath)
        {
            string[] lines = File.ReadAllLines(filePath); ;
            double[][] matrix = new double[lines.Length][];

            for (int i = 0; i < lines.Length; i++)
            {
                string[] rowValues = lines[i].Split(',');
                // Split each line into an array of strings (columns)
                matrix[i] = rowValues
                    .Take(6)
                    .Select(double.Parse)
                    .ToArray();
            }
            return matrix;
        }

        private double[][] RetrieveOutputDataFromCSV(string filePath)
        {
            string[] lines = File.ReadAllLines(filePath); ;
            double[][] matrix = new double[lines.Length][];

            for (int i = 0; i < lines.Length; i++)
            {
                string[] rowValues = lines[i].Split(',');
                // Split each line into an array of strings (columns)
                matrix[i] = rowValues
                    .Skip(Math.Max(0, rowValues.Length - 4))
                    .Select(double.Parse)
                    .ToArray();
            }
            return matrix;
        }

        // function to train the model with the training data
        public void TrainModel()
        {
            for (int epoch = 0; epoch < _epochs; epoch++)
            {
                double totalError = 0;
                double CountWrong = 0.0;

                for (int i = 0; i < trainingInputs.Length; i++)
                {
                    // stores the weights of the hiddenlayer inputs
                    double[] hiddenLayerInput = AddBias(MatrixVectorMultiply(_network.hiddenWeights, trainingInputs[i]), _network.hiddenBias);

                    // stores the output of all hiddenlayer neurons
                    double[] hiddenLayerOutput = ApplyActivationFunction(hiddenLayerInput);

                    // stores the weights of the outputlayer inputs
                    double[] outputLayerInput = AddBias(MatrixVectorMultiply(_network.outputWeights, hiddenLayerOutput), _network.outputBias);

                    // stores the output of all outputlayer neurons
                    double[] output = ApplyActivationFunction(outputLayerInput);

                    double maxValueIst = output.Max();
                    double IndexIst = output.ToList().IndexOf(maxValueIst);
                    double maxValueSoll = trainingOutputs[i].Max();
                    double IndexSoll = trainingOutputs[i].ToList().IndexOf(maxValueSoll);

                    if (IndexIst != IndexSoll)
                    {
                        CountWrong++;
                        if (epoch == _epochs - 1)
                        {
                            Console.WriteLine("\nFalsch Interpretierter Datensatz:");
                            Console.WriteLine("         \t Empf\tMotiv\tFach\tBBV\tMathe\tEngli\tDeutsch");
                            Console.WriteLine($"TrainInputs:\t {trainingInputs[i][0]:F3}\t{trainingInputs[i][1]:F3}\t{trainingInputs[i][2]:F3}\t{trainingInputs[i][3]:F3}\t{trainingInputs[i][4]:F3}\t{trainingInputs[i][5]:F3}");
                            Console.WriteLine("\n          \t TOP\tA  \tB  \tC   ");
                            Console.WriteLine($"IstOutput :\t {output[0]:F3}\t{output[1]:F3}\t{output[2]:F3}\t{output[3]:F3}");
                            Console.WriteLine($"SollOutput:\t {trainingOutputs[i][0]:F3}\t{trainingOutputs[i][1]:F3}\t{trainingOutputs[i][2]:F3}\t{trainingOutputs[i][3]:F3}\n");
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
                            hiddenDeltas[j] += outputDeltas[k] * _network.outputWeights[j, k];
                        }
                        hiddenDeltas[j] *= hiddenLayerOutput[j] * (1 - hiddenLayerOutput[j]);
                    }

                    // Aktualisierung der Gewichte und Biases im Output-Layer
                    for (int j = 0; j < hiddenLayerOutput.Length; j++)
                    {
                        for (int k = 0; k < output.Length; k++)
                        {
                            _network.outputWeights[j, k] += _learningRate * outputDeltas[k] * hiddenLayerOutput[j];
                        }
                    }
                    for (int j = 0; j < _network.outputBias.Length; j++)
                    {
                        _network.outputBias[j] += _learningRate * outputDeltas[j];
                    }

                    // Aktualisierung der Gewichte und Biases im Hidden-Layer
                    for (int j = 0; j < _network.hiddenWeights.GetLength(0); j++)
                    {
                        for (int k = 0; k < _network.hiddenWeights.GetLength(1); k++)
                        {
                            _network.hiddenWeights[j, k] += _learningRate * hiddenDeltas[k] * trainingInputs[i][j];
                        }
                    }
                    for (int j = 0; j < _network.hiddenBias.Length; j++)
                    {
                        _network.hiddenBias[j] += _learningRate * hiddenDeltas[j];
                    }
                }

                double wrongPercent = CountWrong / trainingInputs.Length * 100.0;
                _totalError = totalError;
                _wrongPercentage = wrongPercent;

                // Ausgabe des Fehlers nach jeder Epoche
                Console.WriteLine($"Epoch {epoch + 1}\t Error: {totalError:F4}\t Fehlerprozent: {wrongPercent:F2}");
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

    }
}
