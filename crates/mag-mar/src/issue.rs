//! The pool's authority and its members' certificates, as the host makes them.
//!
//! Each pool has its own authority. The host keeps its key, and gives each member a certificate
//! signed by it that names the member, with a key of its own. A member that joins later, or
//! comes back after its lease ended, is given a new certificate by the same authority, and the
//! coordinator is told nothing: the certificate proves the member.
//!
//! Every certificate has a new random serial, which [`crate::Identity`] reports, so that a
//! certificate issued after a lease ended can be told from the one it ended on. A member's
//! certificate names it as a DNS name, which [`crate::check_name`] limits to lower case, is
//! good for both ends of a link (TLS client and server), and cannot sign another certificate.
//! The keys are ECDSA on P-256.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use rcgen::{
    BasicConstraints, CertificateParams, DistinguishedName, DnType, ExtendedKeyUsagePurpose, IsCa,
    Issuer, KeyPair, KeyUsagePurpose, SerialNumber,
};
use ring::rand::SecureRandom;
use time::OffsetDateTime;

use crate::{Credentials, LinkError, check_name};

/// The authority's common name. The certificate is trusted for its key, not for this name.
const AUTHORITY_NAME: &str = "mag pool authority";

/// How long before now a certificate is valid from, for clocks a little behind the host's.
const SKEW: Duration = Duration::from_secs(5 * 60);

/// A pool's certificate authority, with its private key: the host's alone.
pub struct Authority {
    certificate: String,
    issuer: Issuer<'static, KeyPair>,
}

/// A member's certificate and private key, in PEM, as issued.
pub struct Issued {
    pub member: String,
    pub certificate: String,
    pub key: String,
    /// The certificate's serial, as [`crate::Identity`] reports it.
    pub serial: Vec<u8>,
}

fn failed(e: rcgen::Error) -> LinkError {
    LinkError::Credentials(e.to_string())
}

fn at(t: SystemTime) -> Result<OffsetDateTime, LinkError> {
    let secs = t
        .duration_since(UNIX_EPOCH)
        .map_err(|e| LinkError::Credentials(e.to_string()))?
        .as_secs();
    OffsetDateTime::from_unix_timestamp(secs as i64)
        .map_err(|e| LinkError::Credentials(e.to_string()))
}

/// Valid from a little before now, for `days`.
fn validity(params: &mut CertificateParams, days: u32) -> Result<(), LinkError> {
    let now = SystemTime::now();
    params.not_before = at(now - SKEW)?;
    params.not_after = at(now + Duration::from_secs(u64::from(days) * 24 * 60 * 60))?;
    Ok(())
}

/// A new random serial of 16 bytes, positive and of that full length.
fn serial() -> Result<SerialNumber, LinkError> {
    let mut bytes = [0; 16];
    ring::rand::SystemRandom::new()
        .fill(&mut bytes)
        .map_err(|_| LinkError::Credentials("no random bytes for a serial".into()))?;
    bytes[0] = (bytes[0] & 0x7f) | 0x40;
    Ok(SerialNumber::from_slice(&bytes))
}

impl Authority {
    /// A new authority for a pool, valid for `days`. It signs members' certificates only.
    pub fn new(days: u32) -> Result<Authority, LinkError> {
        let mut params = CertificateParams::default();
        let mut name = DistinguishedName::new();
        name.push(DnType::CommonName, AUTHORITY_NAME);
        params.distinguished_name = name;
        params.is_ca = IsCa::Ca(BasicConstraints::Constrained(0));
        params.key_usages = vec![KeyUsagePurpose::KeyCertSign, KeyUsagePurpose::CrlSign];
        params.serial_number = Some(serial()?);
        validity(&mut params, days)?;
        let key = KeyPair::generate().map_err(failed)?;
        let certificate = params.self_signed(&key).map_err(failed)?.pem();
        Ok(Authority {
            certificate,
            issuer: Issuer::new(params, key),
        })
    }

    /// An authority made before, from its certificate and its key, in PEM.
    pub fn from_pem(certificate: &str, key: &str) -> Result<Authority, LinkError> {
        let key = KeyPair::from_pem(key).map_err(failed)?;
        let issuer = Issuer::from_ca_cert_pem(certificate, key).map_err(failed)?;
        Ok(Authority {
            certificate: certificate.into(),
            issuer,
        })
    }

    /// The authority's certificate, in PEM: public, and what every start file of the pool
    /// holds.
    pub fn certificate(&self) -> &str {
        &self.certificate
    }

    /// The authority's private key, in PEM: the host keeps it, and gives it to no member.
    pub fn key(&self) -> String {
        self.issuer.key().serialize_pem()
    }

    /// Issues `member` a certificate naming it, with a new key and a new serial, valid from now
    /// for `days`; and checks it as a member's link will.
    pub fn issue(&self, member: &str, days: u32) -> Result<Issued, LinkError> {
        check_name(member)?;
        let mut params = CertificateParams::new(vec![member.to_string()]).map_err(failed)?;
        let mut name = DistinguishedName::new();
        name.push(DnType::CommonName, member);
        params.distinguished_name = name;
        params.is_ca = IsCa::ExplicitNoCa;
        params.key_usages = vec![KeyUsagePurpose::DigitalSignature];
        params.extended_key_usages = vec![
            ExtendedKeyUsagePurpose::ServerAuth,
            ExtendedKeyUsagePurpose::ClientAuth,
        ];
        params.use_authority_key_identifier_extension = true;
        params.serial_number = Some(serial()?);
        validity(&mut params, days)?;
        let key = KeyPair::generate().map_err(failed)?;
        let certificate = params.signed_by(&key, &self.issuer).map_err(failed)?.pem();
        let key = key.serialize_pem();
        let credentials = Credentials::from_pem(member, &self.certificate, &certificate, &key)?;
        Ok(Issued {
            member: member.into(),
            certificate,
            key,
            serial: credentials.identity().serial,
        })
    }
}
