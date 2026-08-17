package com.example.tidapp

import android.content.Intent
import android.os.Bundle
import android.util.Patterns
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.google.firebase.auth.FirebaseAuth

class LoginActivity : AppCompatActivity() {

    private lateinit var auth: FirebaseAuth

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // 1. Firebase örneğini hemen başlatıyoruz
        auth = FirebaseAuth.getInstance()

        // 🌟 OTOMATİK GİRİŞ KONTROLÜ (BENİ HATIRLA) 🌟
        // Eğer cihazda daha önce başarıyla giriş yapmış bir kullanıcı oturumu varsa
        if (auth.currentUser != null) {
            // Arayüzü dahi yüklemeden direkt Ana Menüye yönlendiriyoruz reisim
            val intent = Intent(this, MenuActivity::class.java)
            startActivity(intent)
            finish() // Bu sayfayı kapatıyoruz ki geri tuşuna basınca buraya düşmesin
            return // Alt taraftaki kodların çalışmasını engellemek için onCreate'i burada bitiriyoruz
        }

        enableEdgeToEdge()
        setContentView(R.layout.activity_login)

        val inputField = findViewById<EditText>(R.id.etEmail)
        val passwordField = findViewById<EditText>(R.id.etPassword)
        val loginButton = findViewById<Button>(R.id.btnLogin)
        val goToRegisterText = findViewById<TextView>(R.id.tvGoToRegister)

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        goToRegisterText.setOnClickListener {
            val intent = Intent(this, RegisterActivity::class.java)
            startActivity(intent)
        }

        loginButton.setOnClickListener {
            val usernameOrEmail = inputField.text.toString().trim()
            val password = passwordField.text.toString().trim()

            if (usernameOrEmail.isNotEmpty() && password.isNotEmpty()) {
                if (Patterns.EMAIL_ADDRESS.matcher(usernameOrEmail).matches()) {
                    loginWithEmail(usernameOrEmail, password)
                } else {
                    val simulatedEmail = "$usernameOrEmail@gmail.com"
                    loginWithEmail(simulatedEmail, password)
                }
            } else {
                Toast.makeText(this, "Lütfen tüm alanları doldurun", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun loginWithEmail(email: String, password: String) {
        auth.signInWithEmailAndPassword(email, password)
            .addOnCompleteListener(this) { task ->
                if (task.isSuccessful) {
                    Toast.makeText(this, "Giriş Başarılı!", Toast.LENGTH_SHORT).show()

                    val intent = Intent(this, MenuActivity::class.java)
                    startActivity(intent)
                    finish()
                } else {
                    Toast.makeText(this, "Hatalı şifre veya kullanıcı adı/e-posta!", Toast.LENGTH_LONG).show()
                }
            }
    }
}