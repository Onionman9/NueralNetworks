﻿using System;

/*
    Author: Gage Glenn and Thang Nguyen
 */

namespace NueralNetworks
{
    class Program
    {
        static void Main(string[] args)
        {
            int[][] training_example = new int[][]
            {
                new int[5] { 0,0,0,0,0 },
                new int[5] { 0,0,0,0,1 },
                new int[5] { 0,0,0,1,0 },
                new int[5] { 0,0,1,0,0 },
                new int[5] { 0,1,0,0,0 },
                new int[5] { 0,0,0,1,1 },
                new int[5] { 0,0,1,1,1 },
                new int[5] { 0,1,1,1,1 },
                new int[5] { 0,1,0,1,0 },
                new int[5] { 0,0,1,0,1 },
                new int[5] { 0,1,0,0,1 },
                new int[5] { 0,1,0,1,1 },
                new int[5] { 0,1,1,1,0 },

                new int[5] { 1,0,0,0,0 },
                new int[5] { 1,0,0,0,1 },
                new int[5] { 1,0,0,1,0 },
                new int[5] { 1,0,1,0,0 },
                new int[5] { 1,1,0,0,0 },
                new int[5] { 1,1,0,0,1 },
                new int[5] { 1,1,0,1,0 },
                new int[5] { 1,1,1,0,0 },
                new int[5] { 1,1,1,0,1 },
                new int[5] { 1,1,1,1,0 },
                new int[5] { 1,1,1,1,1 },

            };
            int[][] corresponding_output = new int[][]
            {
                new int[] {0},
                new int[] {1},
                new int[] {1},
                new int[] {1},
                new int[] {1},
                new int[] {0},
                new int[] {1},
                new int[] {0},
                new int[] {0},
                new int[] {0},
                new int[] {0},
                new int[] {1},
                new int[] {1},

                new int[] {1},
                new int[] {0},
                new int[] {0},
                new int[] {0},
                new int[] {0},
                new int[] {1},
                new int[] {1},
                new int[] {1},
                new int[] {0},
                new int[] {0},
                new int[] {1},
            };
            NueralNet myNetwork = new NueralNet(5, 3, 1);
            myNetwork.train_the_network(training_example, corresponding_output);
            int[][] test_case = new int[][]
            {
            new int[] { 1,0,1,0,1 },
            new int[] { 0,0,0,0,0 },
            new int[] { 0,0,1,1,0 },
            new int[] { 1,1,1,1,1 },
            new int[] { 1,1,1,1,0 },
            new int[] { 0,0,1,1,1 },
            new int[] { 1,0,1,1,1 }
            };
            myNetwork.check_Learned_network(test_case);

        }
    }

    class NueralNet 
    {
        double LearningRate = 0.5;
        double MomentumRate = 0.1;
        
        int inputCount;
        int hiddenCount;
        int outputCount;

        double[] activeInput;
        double[] activeHidden;
        double[] activeOutput;

        double[,] weightArrayIn;
        double[,] weightArrayOut;

        double[,] changeIn;
        double[,] changeOut;
        /*
            Nueral network constructor
         */
        public NueralNet(int _inputCount, int _hiddenCount, int _outputCount) 
        {
            inputCount = _inputCount + 1;
            hiddenCount = _hiddenCount;
            outputCount = _outputCount;
            // inputWeights = _inputWeights;

            activeInput = new double[inputCount];
            activeHidden = new double[hiddenCount];
            activeOutput = new double[outputCount];

            weightArrayIn = genArray(inputCount, hiddenCount);      // weights for Input
            weightArrayOut = genArray(hiddenCount, outputCount);    // Weights for Output

            changeIn = genArray(inputCount, hiddenCount);
            changeOut = genArray(hiddenCount, outputCount);
        }
        /*
            Update the values within our network
         */
        public double[] Update(int[] inputs)
        {
            if (inputs.Length != inputCount - 1)
            {
                throw new Exception("NueralNet/update: Invalid Input array size!");
            }

            for (int x = 0; x < inputCount - 1; x++)
            {
                activeInput[x] = inputs[x];
            }

            for (int x = 0; x < hiddenCount; x++)
            {
                double sum = 0.0;
                for (int y = 0; y < inputCount; y++)
                {
                    sum += (activeInput[x] * weightArrayIn[y, x]);
                }
                activeHidden[x] = Sigmoid(sum);
            }

            for (int x = 0; x < outputCount; x++)
            {
                double sum = 0.0;
                for (int y = 0; y < hiddenCount; y++)
                {
                    sum += (activeHidden[y] * weightArrayOut[y, x]);
                }
                activeOutput[x] = Sigmoid(sum);
            }
            return activeOutput;
        }

        /*
            Back Propigate and return the error as a double;
         */

        public double BackPropagate(double learning_rate, double momentum, int[] the_target) 
        {
            if(the_target.Length != this.outputCount)
            {
                throw new Exception("NueralNet/update: Invalid Output array size!");
            }

            double[] error_out_layer = new double[outputCount];
            for(int i = 0; i < outputCount; i ++)
            {
                double temp_error = the_target[i] - activeOutput[i];
                error_out_layer[i] = SigmoidDerivative(activeOutput[i]) * temp_error;
            }

            double[] error_hidden_layer = new double[hiddenCount];
            for(int n = 0; n < hiddenCount; n++)
            {
                double temp_error = 0.0;
                for (int m = 0; m < outputCount; m++)
                {
                    temp_error += (error_out_layer[m] * weightArrayOut[n, m]);
                }
                error_hidden_layer[n] = SigmoidDerivative(activeHidden[n]) * temp_error;
            }

            /*
             * Updating the input and output weights array
             */
            for(int a = 0; a < hiddenCount; a++)
            {
                for(int b = 0; b < outputCount; b++)
                {
                    double the_different = error_out_layer[b] * activeHidden[a];
                    //weightArrayOut[a, b] = weightArrayOut[a, b] + (learning_rate * the_different) + (momentum * changeOut[a, b]);
                    weightArrayOut[a, b] += (learning_rate * the_different) + (momentum * changeOut[a, b]);
                    changeOut[a, b] = the_different;
                }
            }

            for (int c = 0; c < inputCount; c++)
            {
                for (int d = 0; d < hiddenCount; d++)
                {
                    double different_ = error_hidden_layer[d] * activeInput[c];
                    //weightArrayIn[c, d] = weightArrayIn[c, d] + (learning_rate * different_) + (momentum * changeIn[c, d]);
                    weightArrayIn[c, d] += (learning_rate * different_) + (momentum * changeIn[c, d]);
                    changeIn[c, d] = different_;
                }
            }

            double actual_error = 0.0;
            for (int e = 0; e < the_target.Length; e++)
            {
                //Math.Pow(base, power)
                actual_error += 0.5 * Math.Pow((the_target[e] - activeOutput[e]), 2);
            }
            return actual_error;
        }
        /*
         Generates an Array and populates it with new weights between 0 and 1
         */
        private double[,] genArray(int row, int col)
        {
            Random r = new Random();
            double[,] temp_arr = new double[row, col];
            for (int i = 0; i < row; i++)
            {
                for (int j = 0; j < col; j++)
                {
                    temp_arr[i, j] = r.NextDouble();
                }
            }
            return temp_arr;
        }
        /*
            Define the Sigmoid Function
         */
        private static double Sigmoid(double x)
        {
            return (1.0 / (1.0 + Math.Exp((-1 * x))));
        }
        /*
        Define the Sigmoid Derivation Function
         */
        private static double SigmoidDerivative(double x)
        {
            return (Sigmoid(x) * (1 - Sigmoid(x)));
        }
        int[][] arr = new int[2][];

        /*
         Nueral Net method
         */
        public void check_Learned_network(int[][] input_5_bits)
        {
            double[] parity;
            for (int m = 0; m < input_5_bits.Length; m++)
            {
                parity = Update(input_5_bits[m]);

                printStuff(input_5_bits[m]);
                Console.Write(parity[0]);
                foreach (int elem in parity)
                {
                    Console.Write(elem);
                }
                Console.WriteLine();
            }
        }

        private void printStuff(int[] in_Array)
        {
            for (int i = 0; i < in_Array.Length; i++)
            {
                Console.Write(in_Array[i]);
            }
            Console.Write(" :\t");
        }

        public void train_the_network(int[][] train_example, int[][] target)
        {
            // or we could do while error > 0.005 
            for (int f = 0; f < 200; f++)
            {
                double network_err = 0.0;
                for (int g = 0; g < train_example.Length; g++)
                {
                    int[] input = train_example[g];
                    int[] expected = target[g];

                    this.Update(input);
                    network_err += this.BackPropagate(LearningRate, MomentumRate, expected);
                }

                if (f % 10 == 0)
                {
                    Console.WriteLine("Percent Error: " + network_err);
                }
            }
        }
    }
}
