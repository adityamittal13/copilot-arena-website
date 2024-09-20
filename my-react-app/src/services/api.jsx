const BASE_URL = '/api'; // Adjust according to your backend URL

// Function to fetch model leaderboard data
export const fetchModelLeaderboardData = async () => {
  try {
    const response = await fetch(`${BASE_URL}/models/leaderboard`);
    const data = await response.json();
    return data;
  } catch (error) {
    throw new Error('Failed to fetch model leaderboard data');
  }
};

// Function to fetch player leaderboard data
export const fetchPlayerLeaderboardData = async () => {
  try {
    const response = await fetch(`${BASE_URL}/players/leaderboard`);
    const data = await response.json();
    return data;
  } catch (error) {
    throw new Error('Failed to fetch player leaderboard data');
  }
};
