pipeline {
  agent any
  stages {
    stage('Check') {
      steps {
        sh '''apt-get update
apt-get install -y ruby ruby-dev build-essential
echo \'# Install Ruby Gems to ~/gems\' >> ~/.bashrc
echo \'export GEM_HOME=$HOME/gems\' >> ~/.bashrc
echo \'export PATH=$HOME/gems/bin:$PATH\' >> ~/.bashrc
gem install jekyll:3.6.2 bundler'''
      }
    }
    stage('Build') {
      steps {
        sh '''bundle install
bundle exec jekyll build'''
      }
    }
    stage('Deploy') {
      steps {
        git(url: 'https://github.com/shinjayne/shinjayne.github.io.git', branch: 'master')
      }
    }
  }
}