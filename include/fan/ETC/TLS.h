#pragma once

#include _WITCH_PATH(IO/IO.h)
#include _WITCH_PATH(include/openssl.h)

typedef struct{
	EVP_PKEY *w;
}TLS_pkey_t;
typedef struct{
	X509 *w;
}TLS_cert_t;
typedef struct{
	SSL_CTX *w;
}TLS_ctx_t;

void TLS_pkey_free(TLS_pkey_t *pkey){
	EVP_PKEY_free(pkey->w);
}

void TLS_cert_free(TLS_cert_t *cert){
	X509_free(cert->w);
}

void TLS_ctx_free(TLS_ctx_t *ctx){
	SSL_CTX_free(ctx->w);
}

bool TLS_pkey_generate(TLS_pkey_t *pkey){
	pkey->w = EVP_PKEY_new();
	BIGNUM *bne = BN_new();
	BN_set_word(bne, RSA_F4);
	RSA *rsa = RSA_new();
	RSA_generate_key_ex(rsa, 2048, bne, 0);
	EVP_PKEY_assign(pkey->w, EVP_PKEY_RSA, rsa);
	return 0;
}

bool TLS_cert_generate(TLS_pkey_t *pkey, TLS_cert_t *cert){
	cert->w = X509_new();
	X509_gmtime_adj(X509_get_notBefore(cert->w), 0);
	X509_gmtime_adj(X509_get_notAfter(cert->w), 31536000L);
	X509_set_pubkey(cert->w, pkey->w);
	X509_NAME *name = X509_get_subject_name(cert->w);
	X509_set_issuer_name(cert->w, name);
	X509_sign(cert->w, pkey->w, EVP_sha256());
	return 0;
}

bool TLS_ctx_keycert(TLS_ctx_t *ctx, TLS_pkey_t *pkey, TLS_cert_t *cert){
	ctx->w = SSL_CTX_new(TLS_method());
	if(!ctx->w)
		return 1;
	if(SSL_CTX_use_PrivateKey(ctx->w, pkey->w) != 1){
		TLS_ctx_free(ctx);
		return 1;
	}
	if(SSL_CTX_use_certificate(ctx->w, cert->w) != 1){
		TLS_ctx_free(ctx);
		return 1;
	}
	if(SSL_CTX_check_private_key(ctx->w) != 1){
		TLS_ctx_free(ctx);
		return 1;
	}
	SSL_CTX_set_options(ctx->w, SSL_OP_ALL);
	return 0;
}

bool TLS_ctx_generate(TLS_ctx_t *ctx){
	TLS_pkey_t pkey;
	if(TLS_pkey_generate(&pkey))
		return 1;
	TLS_cert_t cert;
	if(TLS_cert_generate(&pkey, &cert)){
		TLS_pkey_free(&pkey);
		return 1;
	}
	if(TLS_ctx_keycert(ctx, &pkey, &cert)){
		TLS_pkey_free(&pkey);
		TLS_cert_free(&cert);
		return 1;
	}
	return 0;
}

bool TLS_ctx_path(TLS_ctx_t *ctx, const char *key_path, const char *cert_path){
	ctx->w = SSL_CTX_new(TLS_method());
	if(!ctx->w)
		return 1;
	if(SSL_CTX_use_PrivateKey_file(ctx->w, key_path, SSL_FILETYPE_PEM) <= 0){
		TLS_ctx_free(ctx);
		return 1;
	}
	if(SSL_CTX_use_certificate_file(ctx->w, cert_path, SSL_FILETYPE_PEM) <= 0){
		TLS_ctx_free(ctx);
		return 1;
	}
	if(!SSL_CTX_check_private_key(ctx->w)){
		TLS_ctx_free(ctx);
		return 1;
	}
	SSL_CTX_set_options(ctx->w, SSL_OP_ALL);
	return 0;
}

enum{
	TLS_peer_server_e = 0x01
};

enum{
	TLS_state_handshake_e = 0x01
};

typedef struct TLS_peer_t TLS_peer_t;
typedef void (*TLS_cb_t)(TLS_peer_t *, uint32_t, bool);
struct TLS_peer_t{
	SSL *ssl;
	BIO *rbio;
	BIO *wbio;
	TLS_cb_t cb;
};

void _TLS_cb(const SSL *ssl, int ptype, int pval){
	uint32_t state;
	bool val;
	switch(ptype){
		case SSL_CB_HANDSHAKE_DONE:{
			if(pval != 1){
				PR_abort();
			}
			state = TLS_state_handshake_e;
			val = 0;
			goto info;
		}
	}
	return;
	info:{
		TLS_peer_t *peer = (TLS_peer_t *)SSL_get_ex_data(ssl, 0);
		peer->cb(peer, state, val);
	}
}

void _TLS_empty_cb(TLS_peer_t *peer, uint32_t flag, bool b){

}

void TLS_peer_set_cb(TLS_peer_t *peer, TLS_cb_t cb){
	peer->cb = cb;
}

bool TLS_peer_open(TLS_peer_t *peer, TLS_ctx_t *ctx, uint32_t flag){
	peer->rbio = BIO_new(BIO_s_mem());
	peer->wbio = BIO_new(BIO_s_mem());

	peer->ssl = SSL_new(ctx->w);
	SSL_set_bio(peer->ssl, peer->rbio, peer->wbio);

	if(flag & TLS_peer_server_e){
		SSL_set_accept_state(peer->ssl);
	}
	else{ /* client */
		SSL_set_connect_state(peer->ssl);
		int r = SSL_do_handshake(peer->ssl);
		if(r == -1 && SSL_get_error(peer->ssl, r) != SSL_ERROR_WANT_READ){
			SSL_free(peer->ssl);
			return 1;
		}
	}
	SSL_set_ex_data(peer->ssl, 0, peer);
	TLS_peer_set_cb(peer, _TLS_empty_cb);
	SSL_set_info_callback(peer->ssl, _TLS_cb);

	return 0;
}
void TLS_peer_close(TLS_peer_t *peer){
	SSL_free(peer->ssl);
}

IO_ssize_t TLS_inwrite(TLS_peer_t *peer, void *data, IO_size_t size){
	int r = BIO_write(peer->rbio, data, size);
	if(r <= 0){
		int err = SSL_get_error(peer->ssl, r);
		switch(err){
			default:{
				return -1;
			}
		}
	}
	else{
		return r;
	}
}

IO_ssize_t TLS_inread(TLS_peer_t *peer, void *data, IO_size_t size){
	int r = SSL_read(peer->ssl, data, size);
	if(r <= 0){
		int err = SSL_get_error(peer->ssl, r);
		switch(err){
			case SSL_ERROR_WANT_READ:{
				return 0;
			}
			default:{
				return -1;
			}
		}
	}
	else{
		return r;
	}
}

IO_ssize_t TLS_outwrite(TLS_peer_t *peer, void *data, IO_size_t size){
	int r = SSL_write(peer->ssl, data, size);
	if(r <= 0){
		int err = SSL_get_error(peer->ssl, r);
		switch(err){
			default:{
				return -1;
			}
		}
	}
	else{
		return r;
	}
}

IO_ssize_t TLS_outread(TLS_peer_t *peer, void *data, IO_size_t size){
	int r = BIO_read(peer->wbio, data, size);
	if(r <= 0){
		int err = SSL_get_error(peer->ssl, r);
		switch(err){
			default:{
				return -1;
			}
		}
	}
	else{
		return r;
	}
}

IO_ssize_t TLS_outreadsize(TLS_peer_t *peer){
	return BIO_ctrl_pending(peer->wbio);
}
