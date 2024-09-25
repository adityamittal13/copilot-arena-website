import React, { useEffect, useState } from 'react';
import { fetchPlayerLeaderboardData } from '../services/api';
import Player from './Player';

const PlayerLeaderboard = () => {
  const [players, setPlayers] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const getPlayerData = async () => {
      try {
        const data = await fetchPlayerLeaderboardData();
        setPlayers(data);
        setLoading(false);
      } catch (error) {
        console.error('Error fetching player leaderboard data:', error);
        setLoading(false);
      }
    };

    getPlayerData();
  }, []);

  if (loading) {
    return <div>Loading players...</div>;
  }

  return (
    <ul>
      {players.map((player) => (
        <Player key={player.id} player={player} />
      ))}
    </ul>
  );
};

export default PlayerLeaderboard;
