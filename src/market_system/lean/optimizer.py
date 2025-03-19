"""
LEAN Strategy Optimizer

This module provides various optimization approaches for LEAN trading strategies:
1. Grid Search - systematic exploration of parameter combinations
2. Genetic Algorithm - evolutionary optimization inspired by natural selection
3. Bayesian Optimization - probabilistic model-based approach
"""

import os
import json
import logging
import itertools
import random
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Callable, Any, Optional, Union
from datetime import datetime
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import asyncio

# Import LEAN components
from src.market_system.lean.lean_adapter import LeanAdapter, BacktestMode

# Optional imports for advanced optimization methods
try:
    from scipy.optimize import minimize
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern
    from skopt import gp_minimize, Optimizer as SkOptimizer
    from skopt.space import Real, Integer, Categorical
    ADVANCED_OPTIMIZERS_AVAILABLE = True
except ImportError:
    ADVANCED_OPTIMIZERS_AVAILABLE = False
    logging.warning("Advanced optimization libraries not found. Some optimization methods will be unavailable.")

logger = logging.getLogger(__name__)

class Parameter:
    """Base class for optimization parameters."""
    
    def __init__(self, name: str, default_value: Any = None):
        """
        Initialize parameter.
        
        Args:
            name: Parameter name (must match the strategy parameter name)
            default_value: Default value for the parameter
        """
        self.name = name
        self.default_value = default_value
    
    def sample(self) -> Any:
        """Sample a value from the parameter space."""
        return self.default_value
    
    def to_dict(self) -> Dict:
        """Convert parameter to dictionary representation."""
        return {
            'name': self.name,
            'type': self.__class__.__name__,
            'default_value': self.default_value
        }

class DiscreteParameter(Parameter):
    """Parameter with discrete values."""
    
    def __init__(self, name: str, values: List[Any], default_index: int = 0):
        """
        Initialize discrete parameter.
        
        Args:
            name: Parameter name
            values: List of possible values
            default_index: Index of default value in values list
        """
        default_value = values[default_index] if 0 <= default_index < len(values) else None
        super().__init__(name, default_value)
        self.values = values
        self.default_index = default_index
    
    def sample(self) -> Any:
        """Sample a random value from the discrete values."""
        return random.choice(self.values)
    
    def to_dict(self) -> Dict:
        """Convert parameter to dictionary representation."""
        base_dict = super().to_dict()
        base_dict.update({
            'values': self.values,
            'default_index': self.default_index
        })
        return base_dict

class IntegerParameter(Parameter):
    """Integer parameter within a range."""
    
    def __init__(self, name: str, min_value: int, max_value: int, default_value: Optional[int] = None):
        """
        Initialize integer parameter.
        
        Args:
            name: Parameter name
            min_value: Minimum value (inclusive)
            max_value: Maximum value (inclusive)
            default_value: Default value
        """
        if default_value is None:
            default_value = (min_value + max_value) // 2
        super().__init__(name, default_value)
        self.min_value = min_value
        self.max_value = max_value
    
    def sample(self) -> int:
        """Sample a random integer from the range."""
        return random.randint(self.min_value, self.max_value)
    
    def to_dict(self) -> Dict:
        """Convert parameter to dictionary representation."""
        base_dict = super().to_dict()
        base_dict.update({
            'min_value': self.min_value,
            'max_value': self.max_value
        })
        return base_dict

class RealParameter(Parameter):
    """Real-valued parameter within a range."""
    
    def __init__(self, name: str, min_value: float, max_value: float, 
                default_value: Optional[float] = None, step: Optional[float] = None):
        """
        Initialize real-valued parameter.
        
        Args:
            name: Parameter name
            min_value: Minimum value (inclusive)
            max_value: Maximum value (inclusive)
            default_value: Default value
            step: Step size for discretizing the range (None = continuous)
        """
        if default_value is None:
            default_value = (min_value + max_value) / 2
        super().__init__(name, default_value)
        self.min_value = min_value
        self.max_value = max_value
        self.step = step
    
    def sample(self) -> float:
        """Sample a random value from the range."""
        if self.step:
            # Discrete sampling with step
            steps = int((self.max_value - self.min_value) / self.step)
            step_index = random.randint(0, steps)
            return self.min_value + step_index * self.step
        else:
            # Continuous sampling
            return random.uniform(self.min_value, self.max_value)
    
    def to_dict(self) -> Dict:
        """Convert parameter to dictionary representation."""
        base_dict = super().to_dict()
        base_dict.update({
            'min_value': self.min_value,
            'max_value': self.max_value,
            'step': self.step
        })
        return base_dict

class OptimizationResult:
    """
    Container for optimization results.
    
    Attributes:
        best_parameters: Dictionary of best parameters found
        best_score: Best score achieved
        all_results: List of all parameter combinations and scores evaluated
        optimization_method: Name of the optimization method used
        metric: Name of the metric optimized
        run_time: Total run time in seconds
        created_at: Timestamp of when the optimization was completed
    """
    
    def __init__(self, 
                best_parameters: Dict[str, Any], 
                best_score: float,
                all_results: List[Dict[str, Any]],
                optimization_method: str,
                metric: str,
                run_time: float):
        """
        Initialize optimization result.
        
        Args:
            best_parameters: Dictionary of best parameters found
            best_score: Best score achieved
            all_results: List of all parameter combinations and scores evaluated
            optimization_method: Name of the optimization method used
            metric: Name of the metric optimized
            run_time: Total run time in seconds
        """
        self.best_parameters = best_parameters
        self.best_score = best_score
        self.all_results = all_results
        self.optimization_method = optimization_method
        self.metric = metric
        self.run_time = run_time
        self.created_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary representation."""
        return {
            'best_parameters': self.best_parameters,
            'best_score': self.best_score,
            'all_results_count': len(self.all_results),
            'optimization_method': self.optimization_method,
            'metric': self.metric,
            'run_time': self.run_time,
            'created_at': self.created_at.isoformat()
        }
    
    def save(self, file_path: str):
        """
        Save optimization results to a file.
        
        Args:
            file_path: Path to save the results
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Prepare data to save
        data = {
            'best_parameters': self.best_parameters,
            'best_score': float(self.best_score),  # Ensure score is serializable
            'all_results': self.all_results,
            'optimization_method': self.optimization_method,
            'metric': self.metric,
            'run_time': self.run_time,
            'created_at': self.created_at.isoformat()
        }
        
        # Save to file
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Optimization results saved to {file_path}")
    
    @classmethod
    def load(cls, file_path: str) -> 'OptimizationResult':
        """
        Load optimization results from a file.
        
        Args:
            file_path: Path to load the results from
            
        Returns:
            OptimizationResult object
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        result = cls(
            best_parameters=data['best_parameters'],
            best_score=data['best_score'],
            all_results=data['all_results'],
            optimization_method=data['optimization_method'],
            metric=data['metric'],
            run_time=data['run_time']
        )
        
        # Parse created_at if available
        if 'created_at' in data:
            result.created_at = datetime.fromisoformat(data['created_at'])
        
        return result

class StrategyOptimizer:
    """
    Optimizer for LEAN trading strategies.
    
    This class provides methods for optimizing strategy parameters using
    different optimization techniques.
    """
    
    def __init__(self, 
                 algorithm_name: str,
                 parameters: List[Parameter],
                 symbols: List[str],
                 start_date: Union[str, datetime],
                 end_date: Union[str, datetime],
                 metric: str = 'sharpe_ratio',
                 data_folder: Optional[str] = None,
                 results_folder: Optional[str] = None,
                 config_path: Optional[str] = None,
                 maximize: bool = True,
                 initial_capital: float = 100000,
                 resolution: str = 'Daily'):
        """
        Initialize the strategy optimizer.
        
        Args:
            algorithm_name: Name of the algorithm to optimize
            parameters: List of parameters to optimize
            symbols: List of symbols to backtest
            start_date: Start date for backtesting
            end_date: End date for backtesting
            metric: Metric to optimize (sharpe_ratio, total_return, etc.)
            data_folder: Path to data folder
            results_folder: Path to results folder
            config_path: Path to configuration file
            maximize: Whether to maximize (True) or minimize (False) the metric
            initial_capital: Initial capital for backtesting
            resolution: Data resolution for backtesting
        """
        self.algorithm_name = algorithm_name
        self.parameters = parameters
        self.symbols = symbols
        
        # Handle dates
        if isinstance(start_date, str):
            self.start_date = start_date
        else:
            self.start_date = start_date.strftime('%Y-%m-%d')
            
        if isinstance(end_date, str):
            self.end_date = end_date
        else:
            self.end_date = end_date.strftime('%Y-%m-%d')
        
        self.metric = metric
        self.maximize = maximize
        self.initial_capital = initial_capital
        self.resolution = resolution
        
        # Initialize LEAN adapter
        self.lean_adapter = LeanAdapter(
            config_path=config_path,
            mode=BacktestMode.LEAN,
            data_folder=data_folder,
            results_folder=results_folder
        )
        
        # Verify the algorithm exists
        available_algorithms = self.lean_adapter.get_available_algorithms()
        if algorithm_name not in available_algorithms:
            raise ValueError(f"Algorithm '{algorithm_name}' not found. Available algorithms: {', '.join(available_algorithms)}")
        
        # Parameter space for advanced optimizers
        self.parameter_space = []
        if ADVANCED_OPTIMIZERS_AVAILABLE:
            for param in parameters:
                if isinstance(param, IntegerParameter):
                    self.parameter_space.append(Integer(
                        param.min_value, param.max_value, name=param.name
                    ))
                elif isinstance(param, RealParameter):
                    self.parameter_space.append(Real(
                        param.min_value, param.max_value, name=param.name
                    ))
                elif isinstance(param, DiscreteParameter):
                    self.parameter_space.append(Categorical(
                        param.values, name=param.name
                    ))
    
    def _evaluate_parameters(self, params: Dict[str, Any]) -> float:
        """
        Evaluate a set of parameters by running a backtest.
        
        Args:
            params: Dictionary of parameter values
            
        Returns:
            Score based on the specified metric
        """
        try:
            # Run backtest with the specified parameters
            backtest_result = self.lean_adapter.run_backtest(
                algorithm_name=self.algorithm_name,
                symbols=self.symbols,
                start_date=self.start_date,
                end_date=self.end_date,
                initial_capital=self.initial_capital,
                resolution=self.resolution,
                parameters=params
            )
            
            # Extract the specified metric
            if 'metrics' in backtest_result and self.metric in backtest_result['metrics']:
                score = backtest_result['metrics'][self.metric]
            else:
                # Default to total return if metric not found
                logger.warning(f"Metric '{self.metric}' not found in backtest results. Using total_return instead.")
                score = backtest_result['metrics'].get('total_return', 0.0)
            
            # Handle the case where we want to minimize
            if not self.maximize:
                score = -score
            
            logger.info(f"Parameters: {params}, Score: {score}")
            return score
            
        except Exception as e:
            logger.error(f"Error evaluating parameters {params}: {e}")
            # Return a very low score on error
            return -float('inf') if self.maximize else float('inf')
    
    def grid_search(self, n_jobs: int = 1) -> OptimizationResult:
        """
        Perform grid search optimization.
        
        Args:
            n_jobs: Number of parallel jobs to run
            
        Returns:
            OptimizationResult containing the best parameters found
        """
        start_time = time.time()
        
        # Generate parameter grid
        param_grid = []
        param_values = []
        param_names = []
        
        for param in self.parameters:
            if isinstance(param, DiscreteParameter):
                param_values.append(param.values)
            elif isinstance(param, IntegerParameter):
                if param.max_value - param.min_value > 100:
                    # Limit the grid size for large ranges
                    step = max(1, (param.max_value - param.min_value) // 10)
                    values = list(range(param.min_value, param.max_value + 1, step))
                else:
                    values = list(range(param.min_value, param.max_value + 1))
                param_values.append(values)
            elif isinstance(param, RealParameter):
                if param.step:
                    # Discrete steps
                    steps = int((param.max_value - param.min_value) / param.step)
                    values = [param.min_value + i * param.step for i in range(steps + 1)]
                else:
                    # For continuous parameters, sample some values
                    num_samples = 10
                    values = [param.min_value + i * (param.max_value - param.min_value) / (num_samples - 1) for i in range(num_samples)]
                param_values.append(values)
            
            param_names.append(param.name)
        
        # Generate all combinations
        all_combinations = list(itertools.product(*param_values))
        
        logger.info(f"Grid search: testing {len(all_combinations)} parameter combinations")
        
        # Function to evaluate a parameter combination
        def evaluate_combination(combination):
            params = {name: value for name, value in zip(param_names, combination)}
            score = self._evaluate_parameters(params)
            return params, score
        
        # Run evaluations
        results = []
        if n_jobs > 1:
            # Parallel execution
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                futures = [executor.submit(evaluate_combination, combo) for combo in all_combinations]
                for future in as_completed(futures):
                    try:
                        params, score = future.result()
                        results.append({'parameters': params, 'score': score})
                    except Exception as e:
                        logger.error(f"Error in worker: {e}")
        else:
            # Sequential execution
            for combo in all_combinations:
                params, score = evaluate_combination(combo)
                results.append({'parameters': params, 'score': score})
        
        # Find best result
        if self.maximize:
            best_result = max(results, key=lambda x: x['score'])
        else:
            best_result = min(results, key=lambda x: x['score'])
        
        run_time = time.time() - start_time
        
        return OptimizationResult(
            best_parameters=best_result['parameters'],
            best_score=best_result['score'] if self.maximize else -best_result['score'],
            all_results=results,
            optimization_method='grid_search',
            metric=self.metric,
            run_time=run_time
        )
    
    def _create_param_dict(self, individual: List[Any]) -> Dict[str, Any]:
        """Convert a list of parameter values to a dictionary."""
        return {param.name: value for param, value in zip(self.parameters, individual)}
    
    def genetic_algorithm(self, 
                          population_size: int = 20, 
                          generations: int = 10,
                          mutation_rate: float = 0.1,
                          crossover_rate: float = 0.7,
                          elitism: int = 2,
                          n_jobs: int = 1) -> OptimizationResult:
        """
        Optimize parameters using a genetic algorithm.
        
        Args:
            population_size: Size of the population
            generations: Number of generations
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            elitism: Number of top individuals to preserve unchanged
            n_jobs: Number of parallel jobs to run
            
        Returns:
            OptimizationResult containing the best parameters found
        """
        if not 0 <= mutation_rate <= 1:
            raise ValueError("Mutation rate must be between 0 and 1")
        if not 0 <= crossover_rate <= 1:
            raise ValueError("Crossover rate must be between 0 and 1")
        if not elitism >= 0:
            raise ValueError("Elitism must be non-negative")
        
        start_time = time.time()
        
        # Initialize population
        population = []
        for _ in range(population_size):
            individual = [param.sample() for param in self.parameters]
            population.append(individual)
        
        # Results storage
        all_results = []
        best_individual = None
        best_score = float('-inf') if self.maximize else float('inf')
        
        # Helper function to evaluate an individual
        def evaluate_individual(individual):
            params = self._create_param_dict(individual)
            score = self._evaluate_parameters(params)
            return individual, score
        
        # Evaluate initial population
        logger.info("Evaluating initial population")
        evaluated_population = []
        
        if n_jobs > 1:
            # Parallel evaluation
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                futures = [executor.submit(evaluate_individual, ind) for ind in population]
                for future in as_completed(futures):
                    try:
                        individual, score = future.result()
                        evaluated_population.append((individual, score))
                        all_results.append({
                            'parameters': self._create_param_dict(individual),
                            'score': score if self.maximize else -score,
                            'generation': 0
                        })
                    except Exception as e:
                        logger.error(f"Error in worker: {e}")
        else:
            # Sequential evaluation
            for individual in population:
                individual, score = evaluate_individual(individual)
                evaluated_population.append((individual, score))
                all_results.append({
                    'parameters': self._create_param_dict(individual),
                    'score': score if self.maximize else -score,
                    'generation': 0
                })
        
        # Update best individual
        for individual, score in evaluated_population:
            if (self.maximize and score > best_score) or (not self.maximize and score < best_score):
                best_individual = individual
                best_score = score
        
        # Genetic algorithm main loop
        for generation in range(1, generations + 1):
            logger.info(f"Generation {generation} of {generations}")
            
            # Sort population by fitness
            evaluated_population.sort(key=lambda x: x[1], reverse=self.maximize)
            
            # Select elite individuals
            elite = [ind for ind, _ in evaluated_population[:elitism]]
            
            # Create new population with elites
            new_population = elite.copy()
            
            # Selection, crossover, mutation
            while len(new_population) < population_size:
                # Selection (tournament selection)
                tournament_size = 3
                tournament = random.sample(evaluated_population, tournament_size)
                tournament.sort(key=lambda x: x[1], reverse=self.maximize)
                parent1 = tournament[0][0]
                
                tournament = random.sample(evaluated_population, tournament_size)
                tournament.sort(key=lambda x: x[1], reverse=self.maximize)
                parent2 = tournament[0][0]
                
                # Crossover
                if random.random() < crossover_rate:
                    # Single-point crossover
                    crossover_point = random.randint(1, len(self.parameters) - 1)
                    child = parent1[:crossover_point] + parent2[crossover_point:]
                else:
                    # No crossover
                    child = parent1.copy()
                
                # Mutation
                for i in range(len(child)):
                    if random.random() < mutation_rate:
                        child[i] = self.parameters[i].sample()
                
                new_population.append(child)
            
            # Evaluate new population
            population = new_population
            evaluated_population = []
            
            if n_jobs > 1:
                # Parallel evaluation
                with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                    futures = [executor.submit(evaluate_individual, ind) for ind in population]
                    for future in as_completed(futures):
                        try:
                            individual, score = future.result()
                            evaluated_population.append((individual, score))
                            all_results.append({
                                'parameters': self._create_param_dict(individual),
                                'score': score if self.maximize else -score,
                                'generation': generation
                            })
                        except Exception as e:
                            logger.error(f"Error in worker: {e}")
            else:
                # Sequential evaluation
                for individual in population:
                    individual, score = evaluate_individual(individual)
                    evaluated_population.append((individual, score))
                    all_results.append({
                        'parameters': self._create_param_dict(individual),
                        'score': score if self.maximize else -score,
                        'generation': generation
                    })
            
            # Update best individual
            for individual, score in evaluated_population:
                if (self.maximize and score > best_score) or (not self.maximize and score < best_score):
                    best_individual = individual
                    best_score = score
            
            logger.info(f"Best score so far: {best_score if self.maximize else -best_score}")
        
        run_time = time.time() - start_time
        
        return OptimizationResult(
            best_parameters=self._create_param_dict(best_individual),
            best_score=best_score if self.maximize else -best_score,
            all_results=all_results,
            optimization_method='genetic_algorithm',
            metric=self.metric,
            run_time=run_time
        )
    
    def bayesian_optimization(self, 
                             n_calls: int = 50,
                             n_random_starts: int = 10,
                             n_jobs: int = 1) -> OptimizationResult:
        """
        Optimize parameters using Bayesian optimization.
        
        Args:
            n_calls: Number of calls to the objective function
            n_random_starts: Number of random evaluations before using surrogate model
            n_jobs: Number of parallel jobs to run
            
        Returns:
            OptimizationResult containing the best parameters found
        """
        if not ADVANCED_OPTIMIZERS_AVAILABLE:
            raise ImportError("scikit-optimize is required for Bayesian optimization. Install it with: pip install scikit-optimize")
        
        start_time = time.time()
        
        # Parameter space for skopt
        dimensions = self.parameter_space
        param_names = [param.name for param in self.parameters]
        
        # Results storage
        all_results = []
        
        # Objective function for optimization
        def objective(x):
            params = {name: value for name, value in zip(param_names, x)}
            score = self._evaluate_parameters(params)
            all_results.append({'parameters': params, 'score': score if self.maximize else -score})
            return -score if self.maximize else score
        
        # Run optimization
        logger.info(f"Running Bayesian optimization with {n_calls} calls")
        result = gp_minimize(
            objective,
            dimensions,
            n_calls=n_calls,
            n_random_starts=n_random_starts,
            random_state=42,
            n_jobs=n_jobs,
            acq_func="EI"  # expected improvement
        )
        
        # Extract best parameters
        best_params = {name: value for name, value in zip(param_names, result.x)}
        best_score = -result.fun if self.maximize else result.fun
        
        run_time = time.time() - start_time
        
        return OptimizationResult(
            best_parameters=best_params,
            best_score=best_score,
            all_results=all_results,
            optimization_method='bayesian_optimization',
            metric=self.metric,
            run_time=run_time
        )
    
    def optimize(self, 
                method: str = 'grid_search', 
                **kwargs) -> OptimizationResult:
        """
        Optimize strategy parameters using the specified method.
        
        Args:
            method: Optimization method ('grid_search', 'genetic_algorithm', 'bayesian_optimization')
            **kwargs: Additional arguments to pass to the optimization method
            
        Returns:
            OptimizationResult containing the best parameters found
        """
        if method == 'grid_search':
            return self.grid_search(**kwargs)
        elif method == 'genetic_algorithm':
            return self.genetic_algorithm(**kwargs)
        elif method == 'bayesian_optimization':
            return self.bayesian_optimization(**kwargs)
        else:
            raise ValueError(f"Unknown optimization method: {method}")

# Convenience functions

def optimize_strategy(algorithm_name: str,
                     parameters: List[Parameter],
                     symbols: List[str],
                     start_date: Union[str, datetime],
                     end_date: Union[str, datetime],
                     metric: str = 'sharpe_ratio',
                     method: str = 'grid_search',
                     maximize: bool = True,
                     data_folder: Optional[str] = None,
                     results_folder: Optional[str] = None,
                     config_path: Optional[str] = None,
                     save_results: bool = True,
                     save_path: Optional[str] = None,
                     **kwargs) -> OptimizationResult:
    """
    Convenience function to optimize a strategy.
    
    Args:
        algorithm_name: Name of the algorithm to optimize
        parameters: List of parameters to optimize
        symbols: List of symbols to backtest
        start_date: Start date for backtesting
        end_date: End date for backtesting
        metric: Metric to optimize (sharpe_ratio, total_return, etc.)
        method: Optimization method ('grid_search', 'genetic_algorithm', 'bayesian_optimization')
        maximize: Whether to maximize (True) or minimize (False) the metric
        data_folder: Path to data folder
        results_folder: Path to results folder
        config_path: Path to configuration file
        save_results: Whether to save the optimization results
        save_path: Path to save the optimization results (if None, uses default path)
        **kwargs: Additional arguments to pass to the optimization method
        
    Returns:
        OptimizationResult containing the best parameters found
    """
    # Create optimizer
    optimizer = StrategyOptimizer(
        algorithm_name=algorithm_name,
        parameters=parameters,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        metric=metric,
        maximize=maximize,
        data_folder=data_folder,
        results_folder=results_folder,
        config_path=config_path
    )
    
    # Run optimization
    result = optimizer.optimize(method=method, **kwargs)
    
    # Save results if requested
    if save_results:
        if save_path is None:
            # Default path
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_dir = results_folder or 'results/optimization'
            os.makedirs(save_dir, exist_ok=True)
            save_path = f"{save_dir}/{algorithm_name}_{method}_{timestamp}.json"
        
        result.save(save_path)
    
    return result

def load_optimization_result(file_path: str) -> OptimizationResult:
    """
    Load optimization results from a file.
    
    Args:
        file_path: Path to the file containing optimization results
        
    Returns:
        OptimizationResult object
    """
    return OptimizationResult.load(file_path) 