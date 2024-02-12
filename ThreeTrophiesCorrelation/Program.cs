using System;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;
using System.Numerics;

public enum Trophy { One, Two, Three }

public class CorrelatedParlayCalculator
{
    // Probabilities of winning each trophy
    static double[] probabilities = { 0.5, 0.5, 0.5 };

    public static double correl = 0.9;

    // Correlations between pairs of events
    static double[,] correlations = {
        { 1.0000, correl, correl },
        { correl, 1.0000, correl },
        { correl, correl, 1.0000 }
    };

    // Random number generator
    static Random rand = new Random();

    // Function to generate correlated random variables using Cholesky decomposition
    static double[] GenerateCorrelatedRandomVariables()
    {
        // Perform Cholesky decomposition
        var chol = Matrix<double>.Build.DenseOfArray(correlations).Cholesky();

        // Generate independent standard normal random variables
        var standardNormals = MathNet.Numerics.LinearAlgebra.Vector<double>.Build.Dense(correlations.GetLength(0), i => GetNextGaussian());

        // Compute correlated random variables
        var correlatedRandomVariables = chol.Factor.Transpose() * standardNormals;

        return correlatedRandomVariables.ToArray();
    }

    // Function to generate a random number from a standard normal distribution
    static double GetNextGaussian()
    {
        // Box-Muller transform to generate two independent standard normal variables
        double u1 = 1.0 - rand.NextDouble();
        double u2 = 1.0 - rand.NextDouble();
        double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
        return z;
    }

    // Function to simulate parlay outcomes
    static bool[] SimulateParlay()
    {
        bool[] outcomes = new bool[probabilities.Length];
        double[] correlatedRandomVariables = GenerateCorrelatedRandomVariables();

        for (int i = 0; i < probabilities.Length; i++)
        {
            double correlatedProb = probabilities[i] + correlatedRandomVariables[i];
            outcomes[i] = rand.NextDouble() < correlatedProb;
        }
        return outcomes;
    }

    // Function to calculate the joint probability of winning the parlay
    static double CalculateJointProbability(int numSimulations)
    {
        int successfulParlays = 0;
        for (int i = 0; i < numSimulations; i++)
        {
            bool[] outcomes = SimulateParlay();
            if (Array.TrueForAll(outcomes, o => o))
            {
                successfulParlays++;
            }
        }
        return (double)successfulParlays / numSimulations;
    }

    public static void Main(string[] args)
    {
        int numSimulations = 1000000;
        double jointProbability = CalculateJointProbability(numSimulations);
        double odds = 1 / jointProbability;

        Console.WriteLine("Joint Probability: " + jointProbability);
        Console.WriteLine("Odds of Winning Parlay: " + odds);
    }
}
