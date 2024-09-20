import React from 'react';
import ModelLeaderboard from './ModelLeaderboard.jsx';
import PlayerLeaderboard from './PlayerLeaderboard.jsx';
import '../App.css';

const Leaderboard = () => {
  return (
    <div className="leaderboard-container">
      <div className="leaderboard models">
        <h1>Model Leaderboard</h1>
        <ModelLeaderboard />
      </div>
      <div className="leaderboard players">
        <h1>Player Leaderboard</h1>
        <PlayerLeaderboard />
      </div>
    </div>
  );
};

export default Leaderboard;
