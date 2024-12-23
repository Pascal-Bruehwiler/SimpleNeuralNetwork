namespace MyNeuralNetwork
{

    public class Network
    {
        public static readonly string projectDirectory = Directory.GetParent(Environment.CurrentDirectory).Parent.Parent.FullName;

        // Attributes of the Network class
        public int network_ID;
        public int inputNeurons;
        public int outputNeurons;
        public int hiddenNeurons;
        public double[,] hiddenWeights;
        public double[] hiddenBias;
        public double[,] outputWeights;
        public double[] outputBias;
        public double NetworkError;
        public double NetworkWrongPercentage;

        // Constructor
        public Network(int input, int hidden, int output)
        {
            inputNeurons = input;
            hiddenNeurons = hidden;
            outputNeurons = output;

            // Generate random weights and bias
            hiddenWeights = GenerateRandomMatrix(inputNeurons, hiddenNeurons);
            outputWeights = GenerateRandomMatrix(hiddenNeurons, outputNeurons);
            hiddenBias = GenerateRandomArray(hiddenNeurons);
            outputBias = GenerateRandomArray(outputNeurons);
        }

        // Function to fill Matrix with random numbers
        private double[,] GenerateRandomMatrix(int rows, int cols)
        {
            Random rand = new Random();
            double[,] matrix = new double[rows, cols];
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    matrix[i, j] = rand.NextDouble();
                }
            }
            return matrix;
        }

        // Function to fill Array with random nubmers
        private double[] GenerateRandomArray(int size)
        {
            Random rand = new Random();
            double[] array = new double[size];
            for (int i = 0; i < size; i++)
            {
                array[i] = rand.NextDouble();
            }
            return array;
        }

        // Function to save weigths and bias to CSV File
        public void SaveNetworkToCSV()
        {
            // Write HiddenWeigths to CSV
            string Path = $"{projectDirectory}\\Data\\HiddenWeights.csv";
            using (StreamWriter outputFile = new StreamWriter(Path))
            {
                for (int i = 0; i < hiddenWeights.GetLength(0); i++)
                {
                    double[] row = new double[hiddenWeights.GetLength(1)];
                    for (int j = 0; j < hiddenWeights.GetLength(1); j++)
                    {
                        row[j] = hiddenWeights[i, j];
                    }
                    string result = string.Join(",", row); // Join each row
                    outputFile.WriteLine(result);
                }
            }

            // Write OutputWeigths to CSV
            Path = $"{projectDirectory}\\Data\\OutputWeights.csv";
            using (StreamWriter outputFile = new StreamWriter(Path))
            {
                for (int i = 0; i < outputWeights.GetLength(0); i++)
                {
                    double[] row = new double[outputWeights.GetLength(1)];
                    for (int j = 0; j < outputWeights.GetLength(1); j++)
                    {
                        row[j] = outputWeights[i, j];
                    }
                    string result = string.Join(",", row); // Join each row
                    outputFile.WriteLine(result);
                }
            }

            // Write HiddenBias to CSV
            Path = $"{projectDirectory}\\Data\\HiddenBias.csv";
            using (StreamWriter outputFile = new StreamWriter(Path))
            {
                foreach (double line in hiddenBias)
                {
                    outputFile.WriteLine(line);
                }
            }

            // Write OutputBias to CSV
            Path = $"{projectDirectory}\\Data\\OutputBias.csv";
            using (StreamWriter outputFile = new StreamWriter(Path))
            {
                foreach (double line in outputBias)
                {
                    outputFile.WriteLine(line);
                }
            }
        }

        // Function to load weigths and bias from CSV File
        public void LoadNetworkFromCSV()
        {
            // Load HiddenWeigths to CSV
            string Path = $"{projectDirectory}\\Data\\HiddenWeights.csv";
            List<double[]> hiddenWeightsList = new List<double[]>();
            foreach (string line in File.ReadLines(Path))
            {
                double[] row = line.Split(',').Select(double.Parse).ToArray();
                hiddenWeightsList.Add(row);
            }
            for (int i = 0; i < hiddenWeightsList.Count; i++)
            {
                for (int j = 0; j < hiddenWeightsList[i].Length; j++)
                {
                    hiddenWeights[i, j] = hiddenWeightsList[i][j];
                }
            }

            // Read OutputWeights from CSV
            Path = $"{projectDirectory}\\Data\\OutputWeights.csv";
            List<double[]> outputWeightsList = new List<double[]>();
            foreach (string line in File.ReadLines(Path))
            {
                double[] row = line.Split(',').Select(double.Parse).ToArray();
                outputWeightsList.Add(row);
            }
            for (int i = 0; i < outputWeightsList.Count; i++)
            {
                for (int j = 0; j < outputWeightsList[i].Length; j++)
                {
                    outputWeights[i, j] = outputWeightsList[i][j];
                }
            }

            // Read HiddenBias from CSV
            Path = $"{projectDirectory}\\Data\\HiddenBias.csv";
            List<double> hiddenBiasList = new List<double>();
            foreach (string line in File.ReadLines(Path))
            {
                hiddenBiasList.Add(double.Parse(line));
            }
            hiddenBias = hiddenBiasList.ToArray();

            // Read OutputBias from CSV
            Path = $"{projectDirectory}\\Data\\OutputBias.csv";
            List<double> outputBiasList = new List<double>();
            foreach (string line in File.ReadLines(Path))
            {
                outputBiasList.Add(double.Parse(line));
            }
            outputBias = outputBiasList.ToArray();
        }

        // Function to print weigths and bias to console
        public void PrintCurrentWeightsBias()
        {
            Console.WriteLine("Hidden Weights");
            Console.WriteLine("----------------");
            for (int i = 0; i < hiddenWeights.GetLength(0); i++) // Loop through rows
            {
                // Create a one-dimensional array for the current row
                string[] row = new string[hiddenWeights.GetLength(1)];
                for (int j = 0; j < hiddenWeights.GetLength(1); j++) // Loop through columns
                {
                    row[j] = hiddenWeights[i, j].ToString();
                }
                Console.WriteLine(string.Join("\t", row)); // Join elements with tabs
            }

            Console.WriteLine("\nOutput Weights");
            Console.WriteLine("----------------");
            for (int i = 0; i < outputWeights.GetLength(0); i++) // Loop through rows
            {
                // Create a one-dimensional array for the current row
                string[] row = new string[outputWeights.GetLength(1)];
                for (int j = 0; j < outputWeights.GetLength(1); j++) // Loop through columns
                {
                    row[j] = outputWeights[i, j].ToString();
                }
                Console.WriteLine(string.Join("\t", row)); // Join elements with tabs
            }

            Console.WriteLine("\nHidden Bias");
            Console.WriteLine("----------------");
            Console.WriteLine(string.Join("\t", hiddenBias));

            Console.WriteLine("\nOutput Bias");
            Console.WriteLine("----------------");
            Console.WriteLine(string.Join("\t", outputBias));
        }
    }
}
