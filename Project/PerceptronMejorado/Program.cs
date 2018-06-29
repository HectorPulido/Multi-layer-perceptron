using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;

namespace PerceptronVideo
{
    class Program
    {
        static string inputPath = @"..\..\..\DataSets\AND.csv";
        static string outputPath = @"..\..\..\DataSets\salida.csv";
        static string neuralNetworkPath = @"..\..\..\DataSets\NN.bin";

        static int inputCount = 2;
        static int outputCount = 1;

        static bool saveNetwork = true;
        static bool loadNetwork = true;

        static double inputMax = 1;
        static double inputMin = 0;

        static double outputMax = 1;
        static double outputMin = 0;

        static List<double[]> input = new List<double[]>();
        static List<double[]> output = new List<double[]>();

        static void ReadData()
        {
            string data = System.IO.File.ReadAllText(inputPath).Replace("\r", "");//.Replace(",", ".");
            string[] row = data.Split(Environment.NewLine.ToCharArray());
            for (int i = 0; i < row.Length; i++)
            {
                string[] rowData = row[i].Split(';');

                double[] inputs = new double[inputCount];
                double[] outputs = new double[outputCount];

                for (int j = 0; j < rowData.Length; j++)
                {
                    if (j < inputCount)
                    {                        
                        inputs[j] = normalize(double.Parse(rowData[j]), inputMin, inputMax);
                       // Console.WriteLine(inputs[j]);
                    }
                    else
                    {
                        outputs[j - inputCount] = normalize(double.Parse(rowData[j]), outputMin, outputMax);
                        //Console.WriteLine(outputs[j - inputCount]);
                    }
                }

                input.Add(inputs);
                output.Add(outputs);
            }

        }


        static double normalize(double val, double min, double max)
        {
            return (val - min) / (max - min);
        }
        static double inverseNormalize(double val, double min, double max)
        {
            return val * (max - min) + min;
        }

        static void Evaluate(Perceptron p, double from, double to, double step)
        { 
            string output = "";
            for (double i = from; i < to; i += step)
            {
                double res = p.Activate(new double[] { normalize(i, inputMin, inputMax) })[0];


                output += i + ";" + inverseNormalize(res, outputMin, outputMax) + "\n";
                Console.WriteLine(i + ";" + res + "\n");
            }

            System.IO.File.WriteAllText(outputPath, output);
        }


        static void Main(string[] args)
        {
            Perceptron p;

            int[] net_def = new int[] { inputCount, 10, 10, outputCount };
            double learning_rate = 0.3;
            double max_error = 0.0001;
            int max_iter = 1000000;

            if (!loadNetwork)
            {
                ReadData();
                p = new Perceptron(net_def);

                while (!p.Learn(input, output, learning_rate, max_error, max_iter, neuralNetworkPath, 10000))
                {
                    p = new Perceptron(net_def);
                }              
            }
            else
            {
                p = Perceptron.Load(neuralNetworkPath);
            }



            //Evaluate(p, 0, 5, 0.1);


            while (true)
            {
                double[] val = new double[inputCount];
                for (int i = 0; i < inputCount; i++)
                {
                    Console.WriteLine("Inserte valor " + i + ": ");
                    val[i] = normalize(double.Parse(Console.ReadLine()), inputMin, inputMax);
                }
                double[] sal = p.Activate(val);
                for (int i = 0; i < outputCount; i++)
                {
                    Console.Write("Respuesta " + i + ": " + inverseNormalize(sal[i],outputMin, outputMax) + " ");
                }
                Console.WriteLine("");
            }

        }
    }
    
    [Serializable]
    class Perceptron
    {
        List<Layer> layers;

        public Perceptron(int[] neuronsPerlayer)
        {
            layers = new List<Layer>();
            Random r = new Random();

            for (int i = 0; i < neuronsPerlayer.Length; i++)
            {
                layers.Add(new Layer(neuronsPerlayer[i], i == 0 ? neuronsPerlayer[i] : neuronsPerlayer[i - 1], r));
            }
        }
        public double[] Activate(double[] inputs)
        {
            double[] outputs = new double[0];
            for (int i = 1; i < layers.Count; i++)
            {
                outputs = layers[i].Activate(inputs);
                inputs = outputs;
            }
            return outputs;
        }
        double IndividualError(double[] realOutput, double[] desiredOutput)
        {
            double err = 0;
            for (int i = 0; i < realOutput.Length; i++)
            {
                err += Math.Pow(realOutput[i] - desiredOutput[i], 2);
            }
            return err;
        }
        double GeneralError(List<double[]> input, List<double[]> desiredOutput)
        {
            double err = 0;
            for (int i = 0; i < input.Count; i++)
            {
                err += IndividualError(Activate(input[i]), desiredOutput[i]);
            }
            return err;
        }
        List<string> log;
        public bool Learn(List<double[]> input, List<double[]> desiredOutput, double alpha, double maxError, int maxIterations, String net_path=null, int iter_save=1)
        {
            double err = 99999;
            log = new List<string>();
            int it = maxIterations;
            while (err > maxError)
            {
                ApplyBackPropagation(input, desiredOutput, alpha);
                err = GeneralError(input, desiredOutput);


                if ((it - maxIterations) % 1000 ==0) {
                    Console.WriteLine(err + " iterations: " + (it - maxIterations));
                }


                if (net_path != null)
                {
                    if ((it - maxIterations) % iter_save == 0)
                    {
                        save_net(net_path);
                        Console.WriteLine("Save net to "+ net_path);
                    }
                }

                log.Add(err.ToString());                
                maxIterations--;

                if (Console.KeyAvailable)
                {
                    System.IO.File.WriteAllLines(@"LogTail.txt", log.ToArray());
                    return true;
                }

                if (maxIterations <= 0)
                {
                    Console.WriteLine("MINIMO LOCAL");
                    System.IO.File.WriteAllLines(@"LogTail.txt", log.ToArray());
                    return false;
                }

            }
            
            System.IO.File.WriteAllLines(@"LogTail.txt", log.ToArray());
            return true;
        }

        List<double[]> sigmas;
        List<double[,]> deltas;

        void SetSigmas(double[] desiredOutput)
        {
            sigmas = new List<double[]>();
            for (int i = 0; i < layers.Count; i++)
            {
                sigmas.Add(new double[layers[i].numberOfNeurons]);
            }
            for (int i = layers.Count - 1; i >= 0; i--)
            {
                for (int j = 0; j < layers[i].numberOfNeurons; j++)
                {
                    if (i == layers.Count - 1)
                    {
                        double y = layers[i].neurons[j].lastActivation;
                        sigmas[i][j] = (Neuron.Sigmoid(y) - desiredOutput[j]) * Neuron.SigmoidDerivated(y);
                    }
                    else
                    {
                        double sum = 0;
                        for (int k = 0; k < layers[i + 1].numberOfNeurons; k++)
                        {
                            sum += layers[i + 1].neurons[k].weights[j] * sigmas[i + 1][k];
                        }
                        sigmas[i][j] = Neuron.SigmoidDerivated(layers[i].neurons[j].lastActivation) * sum;
                    }
                }
            }
        }
        void SetDeltas()
        {
            deltas = new List<double[,]>();
            for (int i = 0; i < layers.Count; i++)
            {
                deltas.Add(new double[layers[i].numberOfNeurons, layers[i].neurons[0].weights.Length]);
            }
        }
        void AddDelta()
        {
            for (int i = 1; i < layers.Count; i++)
            {
                for (int j = 0; j < layers[i].numberOfNeurons; j++)
                {
                    for (int k = 0; k < layers[i].neurons[j].weights.Length; k++)
                    {
                        deltas[i][j, k] += sigmas[i][j] * Neuron.Sigmoid(layers[i - 1].neurons[k].lastActivation);
                    }
                }
            }
        }
        void UpdateBias(double alpha)
        {
            for (int i = 0; i < layers.Count; i++)
            {
                for (int j = 0; j < layers[i].numberOfNeurons; j++)
                {
                    layers[i].neurons[j].bias -= alpha * sigmas[i][j];
                }
            }
        }
        void UpdateWeights(double alpha)
        {
            for (int i = 0; i < layers.Count; i++)
            {
                for (int j = 0; j < layers[i].numberOfNeurons; j++)
                {
                    for (int k = 0; k < layers[i].neurons[j].weights.Length; k++)
                    {
                        layers[i].neurons[j].weights[k] -= alpha * deltas[i][j, k];
                    }
                }
            }
        }
        void ApplyBackPropagation(List<double[]> input, List<double[]> desiredOutput, double alpha)
        {
            SetDeltas();
            for (int i = 0; i < input.Count; i++)
            {
                Activate(input[i]);
                SetSigmas(desiredOutput[i]);
                UpdateBias(alpha);
                AddDelta();
            }
            UpdateWeights(alpha);

        }

        public void save_net(String neuralNetworkPath)
        {
            FileStream fs = new FileStream(neuralNetworkPath, FileMode.Create);
            BinaryFormatter formatter = new BinaryFormatter();
            try
            {
                formatter.Serialize(fs, this);
            }
            catch (SerializationException e)
            {
                Console.WriteLine("Failed to serialize. Reason: " + e.Message);
                throw;
            }
            finally
            {
                fs.Close();
            }
        }

        public static Perceptron Load(String neuralNetworkPath)
        {
            FileStream fs = new FileStream(neuralNetworkPath, FileMode.Open);
            Perceptron p = null;
            try
            {
                BinaryFormatter formatter = new BinaryFormatter();

                // Deserialize the hashtable from the file and 
                // assign the reference to the local variable.
                p = (Perceptron)formatter.Deserialize(fs);
            }
            catch (SerializationException e)
            {
                Console.WriteLine("Failed to deserialize. Reason: " + e.Message);
                throw;
            }
            finally
            {
                fs.Close();
            }

            return p;
        }
    }

    [Serializable]
    class Layer
    {
        public List<Neuron> neurons;
        public int numberOfNeurons;
        public double[] output;

        public Layer(int _numberOfNeurons, int numberOfInputs, Random r)
        {
            numberOfNeurons = _numberOfNeurons;
            neurons = new List<Neuron>();
            for (int i = 0; i < numberOfNeurons; i++)
            {
                neurons.Add(new Neuron(numberOfInputs, r));
            }
        }

        public double[] Activate(double[] inputs)
        {
            List<double> outputs = new List<double>();
            for (int i = 0; i < numberOfNeurons; i++)
            {
                outputs.Add(neurons[i].Activate(inputs));
            }
            output = outputs.ToArray();
            return outputs.ToArray();
        }

    }

    [Serializable]
    class Neuron
    {
        public double[] weights;
        public double lastActivation;
        public double bias;

        public Neuron(int numberOfInputs, Random r)
        {
            bias = 10 * r.NextDouble() - 5;
            weights = new double[numberOfInputs];
            for (int i = 0; i < numberOfInputs; i++)
            {
                weights[i] = 10 * r.NextDouble() - 5;
            }
        }
        public double Activate(double[] inputs)
        {
            double activation = bias;

            for (int i = 0; i < weights.Length; i++)
            {
                activation += weights[i] * inputs[i];
            }

            lastActivation = activation;
            return Sigmoid(activation);
        }
        public static double Sigmoid(double input)
        {
            return 1 / (1 + Math.Exp(-input));
        }
        public static double SigmoidDerivated(double input)
        {
            double y = Sigmoid(input);
            return y * (1 - y);
        }

    }
}
