import '../App.css';
import React from 'react';

const MenuBar = () => {
  return (
    <div className="menu-bar">
      <div className="logo">Code Arena</div>
      <div className="menu-items">
        <a href="#" className="menu-item">Home</a>
        <a href="#" className="menu-item">About</a>
        <a href="#" className="menu-item">Contact</a>
      </div>
    </div>
  );
};

export default MenuBar;
