using System;

namespace NueralNetworks
{
    class Program
    {
        static void Main(string[] args)
        {
            
        }

        static double Sigmoid(double x) 
        {
            return (1.0 / (1.0 + Math.Exp(x)));
        }
    }
}
