pipeline {
  agent any
  stages {
    stage('Check') {
      steps {
        sh 'ls'
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