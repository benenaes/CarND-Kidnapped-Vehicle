/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(
	double x, 
	double y, 
	double theta, 
	double std[]) 
{
	default_random_engine gen;
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	num_particles = 100;

	for (int i = 0; i < num_particles; ++i)
	{
		double sample_x, sample_y, sample_theta;

		sample_x = dist_x(gen);
		sample_y = dist_y(gen);
		sample_theta = dist_theta(gen);

		particles.emplace_back(Particle{ i, sample_x, sample_y, sample_theta, 1, {}, {}, {} });
	}

	is_initialized = true;
}

void ParticleFilter::prediction(
	double delta_t, 
	double std_pos[], 
	double velocity, 
	double yaw_rate) 
{
	default_random_engine gen;
	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);

	for (int i = 0; i < num_particles; ++i)
	{
		if (yaw_rate == 0)
		{
			particles[i].x += velocity * delta_t * cos(particles[i].theta);
			particles[i].y += velocity * delta_t * sin(particles[i].theta);
		}
		else
		{
			particles[i].x += (velocity / yaw_rate) * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
			particles[i].y += (velocity / yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
			particles[i].theta += (yaw_rate * delta_t);
		}

		particles[i].x += dist_x(gen);
		particles[i].y += dist_y(gen);
		particles[i].theta += dist_theta(gen);
	}
}

std::vector<unsigned int> associate_landmarks(
	const Map & map_landmarks, 
	const std::vector<LandmarkObs>& transformed_observations)
{
	std::vector<unsigned int> associations;

	for (unsigned int i = 0; i < transformed_observations.size(); ++i)
	{ 
		double min_distance = numeric_limits<double>::max();
		unsigned int best_index = 0;

		for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); ++j)
		{
			const double map_dist = dist(
				map_landmarks.landmark_list[j].x_f, 
				map_landmarks.landmark_list[j].y_f,
				transformed_observations[i].x,
				transformed_observations[i].y);

			if (map_dist < min_distance)
			{
				best_index = j;
				min_distance = map_dist;
			}
		}

		associations.push_back(best_index);
	}

	return associations;
}

std::vector<LandmarkObs> transform_to_map_coordinates(
	const std::vector<LandmarkObs> & vehicle_observations,
	const Particle & particle)
{
	std::vector<LandmarkObs> transformed_observations;
	for (unsigned int i = 0; i < vehicle_observations.size(); ++i)
	{
		const double trans_x = 
			vehicle_observations[i].x * cos(particle.theta)
			- vehicle_observations[i].y * sin(particle.theta)
			+ particle.x;
		const double trans_y = vehicle_observations[i].y * cos(particle.theta)
			+ vehicle_observations[i].x * sin(particle.theta)
			+ particle.y;
		transformed_observations.emplace_back(LandmarkObs{vehicle_observations[i].id, trans_x, trans_y});
	}

	return transformed_observations;
}

double calculate_single_probability(
	const LandmarkObs & transformed_landmark_observation, 
	const Map::single_landmark_s & associated_landmark,
	const double std_landmark[])
{
	const double sigma_x = std_landmark[0];
	const double sigma_y = std_landmark[1];
	const double denominator = 1 / (2 * M_PI * sigma_x * sigma_y);
	const double exponent = 
		pow(transformed_landmark_observation.x - associated_landmark.x_f, 2) / (2 * pow(sigma_x, 2)) 
		+ pow(transformed_landmark_observation.y - associated_landmark.y_f, 2) / (2 * pow(sigma_y, 2));
	const double prob = exp(-exponent) * denominator;
	return prob;
}

double calculate_multivariate_gaussian_prob(
	const std::vector<LandmarkObs> & transformed_landmark_observations,
	const Map& map,
	const std::vector<unsigned int> & associations,
	const double std_landmark[])
{
	double prob = 1;

	for (unsigned int i = 0; i < transformed_landmark_observations.size(); ++i)
	{
		const double single_prob = calculate_single_probability(transformed_landmark_observations[i], map.landmark_list[associations[i]], std_landmark);
		prob *= single_prob;
	}

	return prob;
}

void ParticleFilter::updateWeights(
	const double sensor_range, 
	const double std_landmark[], 
	const std::vector<LandmarkObs> & observations, 
	const Map & map_landmarks) 
{
	for (unsigned int i = 0; i < particles.size(); ++i)
	{
		const std::vector<LandmarkObs> & transformed_landmarks = transform_to_map_coordinates(observations, particles[i]);
		const std::vector<unsigned int> & associations = associate_landmarks(map_landmarks, transformed_landmarks);
		const double prob = calculate_multivariate_gaussian_prob(transformed_landmarks, map_landmarks, associations, std_landmark);
		particles[i].weight = prob;
	}
}

void ParticleFilter::resample() 
{
	// Resampling wheel implementation

	// Calculate the start index
	default_random_engine gen;
	uniform_int_distribution<> weight_dist(0, num_particles - 1);
	const int start_index = weight_dist(gen);

	// Calculate the maximum weight
	double max_weight = 0.0;
	for (unsigned int i = 0; i < num_particles; ++i)
	{
		if (particles[i].weight > max_weight)
		{
			max_weight = particles[i].weight;
		}
	}

	// Calculate new particle indices
	std::vector<Particle> new_particles;
	double beta = 0;
	std::uniform_real_distribution<> beta_dist(0, 2 * max_weight);

	double index = start_index;

	for (unsigned int i = 0; i < num_particles; ++i)
	{
		beta += beta_dist(gen);

		while (particles[index].weight < beta)
		{
			beta -= particles[index].weight;
			++index;

			if (index == num_particles)
			{
				index = 0;
			}
		}

		new_particles.push_back(particles[index]);
	}

	particles = std::move(new_particles);
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
