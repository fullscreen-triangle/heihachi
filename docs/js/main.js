// Navigation functionality
document.addEventListener('DOMContentLoaded', function() {
    // Smooth scrolling for navigation links
    const navLinks = document.querySelectorAll('a[href^="#"]');
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            const targetSection = document.querySelector(targetId);
            if (targetSection) {
                targetSection.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Tab functionality for features section
    setupTabs('features');
    
    // Tab functionality for output format section
    setupTabs('output-format');
    
    // Mobile navigation toggle
    const hamburger = document.querySelector('.hamburger');
    const navMenu = document.querySelector('.nav-menu');
    
    if (hamburger && navMenu) {
        hamburger.addEventListener('click', function() {
            navMenu.classList.toggle('active');
            hamburger.classList.toggle('active');
        });
    }

    // Generate visualization demo
    generateWaveformDemo();
    generateDrumChart();
});

function setupTabs(sectionId) {
    const section = document.getElementById(sectionId);
    if (!section) return;
    
    const tabButtons = section.querySelectorAll('.tab-button');
    const tabPanes = section.querySelectorAll('.tab-pane');
    
    tabButtons.forEach(button => {
        button.addEventListener('click', function() {
            const targetTab = this.getAttribute('data-tab');
            
            // Remove active class from all buttons and panes
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabPanes.forEach(pane => pane.classList.remove('active'));
            
            // Add active class to clicked button and corresponding pane
            this.classList.add('active');
            const targetPane = section.querySelector(`#${targetTab}`);
            if (targetPane) {
                targetPane.classList.add('active');
            }
        });
    });
}

function generateWaveformDemo() {
    const canvas = document.getElementById('waveform-demo');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    // Clear canvas
    ctx.fillStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.fillRect(0, 0, width, height);
    
    // Generate waveform data
    const points = 100;
    const data = [];
    for (let i = 0; i < points; i++) {
        const x = (i / points) * width;
        const frequency = 0.02 + Math.sin(i * 0.1) * 0.01;
        const amplitude = (Math.sin(i * frequency) + Math.sin(i * frequency * 2.5) * 0.3) * height * 0.3;
        data.push({ x, y: height / 2 + amplitude });
    }
    
    // Draw waveform
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.8)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(data[0].x, data[0].y);
    
    for (let i = 1; i < data.length; i++) {
        ctx.lineTo(data[i].x, data[i].y);
    }
    ctx.stroke();
    
    // Add frequency analysis overlay
    ctx.fillStyle = 'rgba(102, 126, 234, 0.3)';
    for (let i = 0; i < 20; i++) {
        const x = (i / 20) * width;
        const barHeight = Math.random() * height * 0.6 + 20;
        ctx.fillRect(x, height - barHeight, width / 25, barHeight);
    }
}

function generateDrumChart() {
    const canvas = document.getElementById('drumChart');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    
    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Toms', 'Snares', 'Kicks', 'Cymbals', 'Hi-hats'],
            datasets: [{
                data: [15842, 23156, 18934, 21847, 11400],
                backgroundColor: [
                    '#667eea',
                    '#764ba2', 
                    '#f093fb',
                    '#f5576c',
                    '#4facfe'
                ],
                borderWidth: 2,
                borderColor: '#ffffff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        padding: 20,
                        usePointStyle: true,
                        font: {
                            size: 12,
                            family: 'Inter, sans-serif'
                        }
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const total = context.dataset.data.reduce((a, b) => a + b, 0);
                            const percentage = ((context.parsed / total) * 100).toFixed(1);
                            return `${context.label}: ${context.parsed.toLocaleString()} (${percentage}%)`;
                        }
                    }
                }
            }
        }
    });
}

// Add intersection observer for animations
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, observerOptions);

// Observe all sections for scroll animations
document.addEventListener('DOMContentLoaded', function() {
    const sections = document.querySelectorAll('.section');
    sections.forEach(section => {
        section.style.opacity = '0';
        section.style.transform = 'translateY(30px)';
        section.style.transition = 'opacity 0.6s ease-out, transform 0.6s ease-out';
        observer.observe(section);
    });
}); 