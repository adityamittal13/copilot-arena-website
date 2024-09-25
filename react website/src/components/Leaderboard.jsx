import React from 'react';
import ModelLeaderboard from './ModelLeaderboard.jsx';
import PlayerLeaderboard from './PlayerLeaderboard.jsx';
import '../App.css';

const Leaderboard = () => {
  return (
    <div className="leaderboard-wrapper">
      <div className="leaderboard-container">
        <div className="leaderboard-column">
          <h2>Model Leaderboard</h2>
          <ModelLeaderboard />
        </div>
        <div className="leaderboard-column">
          <h2>Player Leaderboard</h2>
          <PlayerLeaderboard />
        </div>
      </div>
    </div>
  );
};

export default Leaderboard;