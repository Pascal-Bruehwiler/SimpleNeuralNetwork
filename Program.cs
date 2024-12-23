
namespace MyNeuralNetwork
{
    class Program
    {
        public static readonly string projectDirectory = Directory.GetParent(Environment.CurrentDirectory).Parent.Parent.FullName;

        public static void Main(string[] args)
        {
            // Initialize new Network
            Network network = new Network(7, 10, 4);

            // Load existing data if available
            try { network.LoadNetworkFromCSV(); }
            catch (Exception ex) { }

            // initialize new training object
            Training newTraining = new Training(network);

            bool isValid = true;
            string currentDateTime = "";

            while (isValid)
            {
                switch (ShowMainMenu())
                {
                    case 1:
                        network.LoadNetworkFromCSV();
                        newTraining.TrainModel();
                        network.SaveNetworkToCSV();
                        network.NetworkError = newTraining._totalError;
                        network.NetworkWrongPercentage = newTraining._wrongPercentage;
                        Console.WriteLine("\nPress any key to go to the main menu.");
                        Console.ReadKey();
                        break;
                    case 2:
                        Console.WriteLine($"New epochs per training (current {newTraining._epochs}): ");
                        try
                        {
                            int newEpoch = int.Parse(Console.ReadLine());
                            newTraining._epochs = newEpoch;
                        }
                        catch (Exception) { Console.WriteLine("Invalid Input"); }
                        Console.WriteLine($"New learning rate (current {newTraining._learningRate}): ");
                        try
                        {
                            double newLearningrate = double.Parse(Console.ReadLine());
                            newTraining._learningRate = newLearningrate;
                        }
                        catch (Exception) { Console.WriteLine("Invalid Input"); }
                        break;
                    case 3:
                        Console.WriteLine("Current Accuracy of the network:");
                        Console.WriteLine($"Total Error:      {newTraining._totalError:F4}");
                        Console.WriteLine($"Wrong Percentage: {newTraining._wrongPercentage:F4}");
                        Console.WriteLine("\nPress any key to go to the main menu.");
                        Console.ReadKey();
                        break;
                    case 4:

                    case 5:
                        currentDateTime = DateTime.Now.ToString("yyyy_dd_MM_h_mm_ss");
                        CopyFilesRecursively($"{projectDirectory}\\Data", $"{projectDirectory}\\OldModels\\{currentDateTime}");
                        break;
                    case 6:

                        break;
                    case 8:
                        currentDateTime = DateTime.Now.ToString("yyyy_dd_MM_h_mm_ss");
                        CopyFilesRecursively($"{projectDirectory}\\Data", $"{projectDirectory}\\OldModels\\{currentDateTime}");
                        network = new Network(7, 10, 4);
                        newTraining = new Training(network);
                        network.SaveNetworkToCSV();
                        network.NetworkError = newTraining._totalError;
                        network.NetworkWrongPercentage = newTraining._wrongPercentage;
                        Console.WriteLine("\nPress any key to go to the main menu.");
                        Console.ReadKey();
                        break;
                    case 9:
                        isValid = false;
                        break;
                    default:
                        Console.WriteLine("Ungültige Auswahl");
                        break;
                }
            }
        }

        public static int ShowMainMenu()
        {
            int option = 0;
            Console.Clear();
            Console.WriteLine(" Main Menu");
            Console.WriteLine(" --------------");
            Console.WriteLine(" 1: Train Existing Model");
            Console.WriteLine(" 2: Change Training Settings");
            Console.WriteLine(" 3: Get Current Accuracy");
            Console.WriteLine(" 4: Get Current Weight/Bias");
            Console.WriteLine(" 5: Store Existing Model");
            Console.WriteLine(" 6: Check Sample Data");
            Console.WriteLine(" 8: Start New Training");
            Console.WriteLine(" 9: Leave Application");
            Console.WriteLine(" --------------");
            Console.WriteLine(" Choose option (number): ");
            string input = Console.ReadLine();
            try
            {
                option = int.Parse(input);
                return option;
            }
            catch (Exception)
            {
                return 0;
                throw;
            }
        }

        private static void CopyFilesRecursively(string sourcePath, string targetPath)
        {
            // Create new folder to store the Files
            Directory.CreateDirectory(targetPath);

            // Copy all the files & Replaces any files with the same name
            foreach (string newPath in Directory.GetFiles(sourcePath, "*.*", SearchOption.AllDirectories))
            {
                File.Copy(newPath, newPath.Replace(sourcePath, targetPath), true);
            }
        }
    }
}
