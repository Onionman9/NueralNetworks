using System;

/*
    Author: Gage Glenn and Thang Nguyen
 */

namespace NueralNetworks
{
    class Program
    {
        static void Main(string[] args)
        {
            // 5 inputs
            int[] training_input = new int[] { 0,0,1,0,1 };

            // 2 outputs.
            int[] training_output = new int[] { 0, 1 };

            // 5 input nodes, 1 hidden layer (contains 2 nodes), and 1 output node indicate even or odd.
            //double[,] weight_in = genArray(5, 2);
            //double[,] weight_out = genArray(2, 1);

            ///*
            // Check first matrix
            // */

            //Console.WriteLine("\n Input_Matrix initial weights \n");
            //for(int m = 0; m < 5; m++)
            //{
            //    for (int n = 0; n < 2; n++)
            //    {
            //        Console.Write(weight_in[m, n] + " ");
            //    }
            //    Console.WriteLine(); 
            //}

            ///*
            //    Check Second matrix
            // */
            //Console.WriteLine("\n Output_Matrix initial weights \n");
            //for (int m = 0; m < 2; m++)
            //{
            //    for (int n = 0; n < 1; n++)
            //    {
            //        Console.Write(weight_out[m, n] + " ");
            //    }
            //    Console.WriteLine();
            //}

            //double learning_rate = 0.05;
            //double momentum = 0.1;
           
        }

        private static double Sigmoid(double x) 
        {
            return (1.0 / (1.0 + Math.Exp(x)));
        }

        private static double SigmoidDerivative(double x)
        {
            return (Sigmoid(x) * (1 - Sigmoid(x)));
        }
        //private static double[,] genArray(int row, int col)
        //{
        //    Random r = new Random();
        //    double[,] temp_arr = new double[row, col]; 
        //    for(int i = 0; i < row; i++)
        //    {
        //        for (int j = 0; j < col; j++)
        //        {
        //            temp_arr[i,j] = r.NextDouble();
        //        }
        //    }
        //    return temp_arr;
        //}
    }

    class NueralNet 
    {
        double LearningRate = 0.05;
        double MomentumRate = 0.1;
        
        int inputCount;
        int hiddenCount;
        int outputCount;
        // int[] inputWeights;

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
        public NueralNet(int _inputCount, int _hiddenCount, int _outputCount/*, int[] _inputWeights*/) 
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
            /*
            for (int x = 0; x < inputCount; x++)
            {
                for (int y = 0; y < hiddenCount; y++)
                {
                    weightArrayIn[x, y] = inputWeights[x];
                }
            }

            for (int x = 0; x < hiddenCount; x++)
            {
                for (int y = 0; y < outputCount; y++)
                {
                    weightArrayIn[x, y] = inputWeights[x + inputCount + outputCount];
                }
            }
            */
            changeIn = genArray(inputCount, hiddenCount);
            changeOut = genArray(hiddenCount, outputCount);
        }
        /*
            Update the values within our network
         */
        public void Update(int[] inputs) 
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
                    sum += (activeHidden[y] * weightArrayIn[y,x]);
                }
                activeOutput[x] = Sigmoid(sum);
            }

            for (int x = 0; x < outputCount; x++)
            {
                double sum = 0.0;
                for (int y = 0; y < inputCount; y++)
                {
                    sum += (activeHidden[y] * weightArrayOut[y, x]);
                }
                activeOutput[x] = Sigmoid(sum);
            }
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
            return (1.0 / (1.0 + Math.Exp(x)));
        }
        /*
        Define the Sigmoid Derivation Function
         */
        private static double SigmoidDerivative(double x)
        {
            return (Sigmoid(x) * (1 - Sigmoid(x)));
        }
    }
}
