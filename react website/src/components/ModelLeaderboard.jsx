import React, { useEffect, useState } from 'react';
import { fetchModelLeaderboardData } from '../services/api';
import Player from './Player'; // Can reuse this for both models and players

const ModelLeaderboard = () => {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const getModelData = async () => {
      try {
        const data = await fetchModelLeaderboardData();
        setModels(data);
        setLoading(false);
      } catch (error) {
        console.error('Error fetching model leaderboard data:', error);
        setLoading(false);
      }
    };

    getModelData();
  }, []);

  if (loading) {
    return <div>Loading models...</div>;
  }

  return (
    <ul>
      {models.map((model) => (
        <Player key={model.id} player={model} />
      ))}
    </ul>
  );
};

export default ModelLeaderboard;
