pipeline {
    agent any // This tells Jenkins it can run on any available server/node

    stages {
        stage('Checkout Code') {
            steps {
                // Jenkins automatically pulls your code from Git here
                checkout scm
            }
        }

        stage('Setup Environment') {
            steps {
                // We use standard terminal commands to set up Python
                bat '''
                python -m venv venv
                call venv\\Scripts\\activate
                pip install -r requirements.txt
                pip install pytest httpx
                '''
            }
        }

        stage('Run Tests') {
            steps {
                // Run the pytest script we created earlier
                bat '''
                call venv\\Scripts\\activate
                pytest test_app.py
                '''
            }
        }
    }
    
    post {
        always {
            echo 'Pipeline execution finished.'
        }
        success {
            echo 'All tests passed! The ML model is ready for deployment.'
        }
        failure {
            echo 'Tests failed. Check the logs before deploying.'
        }
    }
}