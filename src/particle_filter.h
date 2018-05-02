/*
 * particle_filter.h
 *
 * 2D particle filter class.
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#ifndef PARTICLE_FILTER_H_
#define PARTICLE_FILTER_H_

#include "helper_functions.h"

struct Particle {

	int id;
	double x;
	double y;
	double theta;
	double weight;
	std::vector<int> associations;
	std::vector<double> sense_x;
	std::vector<double> sense_y;
};

/**
 * Associate landmark observations (in map coordinates) with landmarks defined in a map
 * @param map_landmarks Map containing all landmark data
 * @param transformed_observations landmark observations (in map coordinates)
 * @return Associations made in between landmark observations and landmark data on the map.
 *  For example: the first element is the index of the landmark in "map" that is associated with the first observed landmark, etc.
*/
std::vector<unsigned int> associate_landmarks(
	const Map & map_landmarks,
	const std::vector<LandmarkObs>& transformed_observations);

/**
 * Transform observations from a particle perspective to map coordinates
 * @param vehicle_observations Observations from a vehicle's (particle's) perspective
 * @param particle A particle
 * @return Observations transformed to map coordinates
*/
std::vector<LandmarkObs> transform_to_map_coordinates(
	const std::vector<LandmarkObs> & vehicle_observations,
	const Particle & particle);

/**
 * Calculate the Gaussian probability (partial weight) of a particle's state given a landmark observation of an existing (known) landmark
 * @param transformed_landmark_observation The observed position of a landmark calculated from a particle's state and a given distance
 * @param associated_landmark Data of the associated map landmark
 * @param std_landmark Standard deviations of landmark observations (dependent on the sensor's technology)
 * @return The partial particle's weight (unimodal Gaussian probability)
*/
double calculate_single_probability(
	const LandmarkObs & transformed_landmark_observation,
	const Map::single_landmark_s & associated_landmark,
	const double std_landmark[]);

/**
 * Calculate the Gaussian probability (weight) of a particle's state given a set of landmark observations of known associated landmarks
 * @param transformed_landmark_observations The observed positions of landmarks calculated from a particle's state and given distances
 * @param map Map containing all landmark data
 * @param associations Associations made in between landmark observations and landmark data on the map. 
 *  For example: the first element is the index of the landmark in "map" that is associated with the first observed landmark, etc.
 * @param std_landmark Standard deviations of landmark observations (dependent on the sensor's technology)
 * @return The multivariate Gaussian probability (weight) of a particle's state
*/
double calculate_multivariate_gaussian_prob(
	const std::vector<LandmarkObs> & transformed_landmark_observations,
	const Map& map,
	const std::vector<unsigned int> & associations,
	const double std_landmark[]);

class ParticleFilter {
	
	// Number of particles to draw
	int num_particles; 
	
	
	
	// Flag, if filter is initialized
	bool is_initialized;
	
	// Vector of weights of all particles
	std::vector<double> weights;
	
public:
	
	// Set of current particles
	std::vector<Particle> particles;

	// Constructor
	// @param num_particles Number of particles
	ParticleFilter() : num_particles(0), is_initialized(false) {}

	// Destructor
	~ParticleFilter() {}

	/**
	 * init Initializes particle filter by initializing particles to Gaussian
	 *   distribution around first position and all the weights to 1.
	 * @param x Initial x position [m] (simulated estimate from GPS)
	 * @param y Initial y position [m]
	 * @param theta Initial orientation [rad]
	 * @param std[] Array of dimension 3 [standard deviation of x [m], standard deviation of y [m]
	 *   standard deviation of yaw [rad]]
	 */
	void init(double x, double y, double theta, double std[]);

	/**
	 * prediction Predicts the state for the next time step
	 *   using the process model.
	 * @param delta_t Time between time step t and t+1 in measurements [s]
	 * @param std_pos[] Array of dimension 3 [standard deviation of x [m], standard deviation of y [m]
	 *   standard deviation of yaw [rad]]
	 * @param velocity Velocity of car from t to t+1 [m/s]
	 * @param yaw_rate Yaw rate of car from t to t+1 [rad/s]
	 */
	void prediction(double delta_t, double std_pos[], double velocity, double yaw_rate);
	
	/**
	 * dataAssociation Finds which observations correspond to which landmarks (likely by using
	 *   a nearest-neighbors data association).
	 * @param predicted Vector of predicted landmark observations
	 * @param observations Vector of landmark observations
	 */
	void dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations);
	
	/**
	 * updateWeights Updates the weights for each particle based on the likelihood of the 
	 *   observed measurements. 
	 * @param sensor_range Range [m] of sensor
	 * @param std_landmark[] Array of dimension 2 [Landmark measurement uncertainty [x [m], y [m]]]
	 * @param observations Vector of landmark observations
	 * @param map Map class containing map landmarks
	 */
	void updateWeights(
		const double sensor_range, 
		const double std_landmark[], 
		const std::vector<LandmarkObs> &observations,
		const Map &map_landmarks);
	
	/**
	 * resample Resamples from the updated set of particles to form
	 *   the new set of particles.
	 */
	void resample();

	/*
	 * Set a particles list of associations, along with the associations calculated world x,y coordinates
	 * This can be a very useful debugging tool to make sure transformations are correct and assocations correctly connected
	 */
	Particle SetAssociations(Particle& particle, const std::vector<int>& associations,
		                     const std::vector<double>& sense_x, const std::vector<double>& sense_y);

	
	std::string getAssociations(Particle best);
	std::string getSenseX(Particle best);
	std::string getSenseY(Particle best);

	/**
	* initialized Returns whether particle filter is initialized yet or not.
	*/
	const bool initialized() const {
		return is_initialized;
	}
};



#endif /* PARTICLE_FILTER_H_ */
